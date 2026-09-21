//! Resident checkpoint registry with preload + LRU eviction.

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};

use crate::engine::DecisionEngine;
use crate::error::Result;
use crate::route::{CheckpointId, RouteDecision};
use crate::router::Router;
use crate::schema::{SystemOneRequest, SystemOneResponse};

use super::checkpoint::CheckpointPaths;
use super::device::DeviceRequest;
use super::neural::NeuralEngine;

/// Lazily loads neural checkpoints and keeps an LRU-capped resident set.
///
/// Routing stays a pure function ([`Router`]); this type owns residency only.
pub struct CheckpointRegistry {
    bundle: PathBuf,
    router: Router,
    device: DeviceRequest,
    max_loaded: usize,
    engines: HashMap<CheckpointId, NeuralEngine>,
    /// Least-recently-used at the front.
    order: VecDeque<CheckpointId>,
}

impl CheckpointRegistry {
    /// Open a checkpoint bundle (english at root or `english/`, plus siblings).
    pub fn open(
        bundle: impl AsRef<Path>,
        device: DeviceRequest,
        max_loaded: usize,
    ) -> Result<Self> {
        let bundle = bundle.as_ref().to_path_buf();
        // Validate at least the default english layout exists.
        let _ = CheckpointPaths::resolve_named(&bundle, CheckpointId::English)?;
        Ok(Self {
            bundle,
            router: Router::default(),
            device,
            max_loaded: max_loaded.max(1),
            engines: HashMap::new(),
            order: VecDeque::new(),
        })
    }

    /// Override the selection router (e.g. enable auto task detection).
    pub fn with_router(mut self, router: Router) -> Self {
        self.router = router;
        self
    }

    /// Bundle root directory.
    pub fn bundle(&self) -> &Path {
        &self.bundle
    }

    /// Current LRU cap.
    pub fn max_loaded(&self) -> usize {
        self.max_loaded
    }

    /// Resident checkpoint ids in LRU order (least recent first).
    pub fn loaded(&self) -> Vec<CheckpointId> {
        self.order.iter().copied().collect()
    }

    /// Raise `max_loaded` and load the given checkpoints up front.
    ///
    /// When `names` is empty, loads english + multilingual + typed-decisions
    /// when each directory is present.
    pub fn preload(&mut self, names: &[CheckpointId]) -> Result<()> {
        let targets: Vec<CheckpointId> = if names.is_empty() {
            CheckpointPaths::present_in_bundle(&self.bundle)
        } else {
            names.to_vec()
        };
        self.max_loaded = self.max_loaded.max(targets.len()).max(1);
        for id in targets {
            let _ = self.load(id)?;
        }
        Ok(())
    }

    /// Attach an already-built engine (bumps `max_loaded` to fit).
    pub fn attach(&mut self, id: CheckpointId, engine: NeuralEngine) {
        self.engines.insert(id, engine);
        self.touch(id);
        self.max_loaded = self.max_loaded.max(self.engines.len()).max(1);
        self.evict();
    }

    /// Return a resident engine, loading from disk on miss.
    pub fn load(&mut self, id: CheckpointId) -> Result<&NeuralEngine> {
        if self.engines.contains_key(&id) {
            self.touch(id);
            return Ok(self.engines.get(&id).expect("just inserted"));
        }
        let paths = CheckpointPaths::resolve_named(&self.bundle, id)?;
        let engine = NeuralEngine::load_with(&paths.root, self.device)?;
        self.engines.insert(id, engine);
        self.order.push_back(id);
        self.evict();
        Ok(self.engines.get(&id).expect("just inserted"))
    }

    /// Borrow a resident engine. Empty when `id` is not loaded.
    pub fn get(&self, id: CheckpointId) -> Option<&NeuralEngine> {
        self.engines.get(&id)
    }

    /// Route then decide with the selected resident engine.
    pub fn system_one(&mut self, request: SystemOneRequest) -> Result<SystemOneResponse> {
        let decision = self.router.route(&request, None, None)?;
        let engine = self.load(decision.model)?;
        engine.decide(&request)
    }

    /// Route then decide, returning the selection and the typed answers.
    ///
    /// The [`RouteDecision`] is the audit record: which checkpoint ran and why.
    /// A missing checkpoint fails closed. The engine does not fall back to a
    /// different pack.
    pub fn system_one_routed(
        &mut self,
        request: SystemOneRequest,
        model: Option<&str>,
        lang: Option<&str>,
    ) -> Result<(RouteDecision, SystemOneResponse)> {
        let decision = self.router.route(&request, model, lang)?;
        let engine = self.load(decision.model)?;
        let response = engine.decide(&request)?;
        Ok((decision, response))
    }

    fn touch(&mut self, id: CheckpointId) {
        self.order.retain(|x| *x != id);
        self.order.push_back(id);
    }

    fn evict(&mut self) {
        while self.order.len() > self.max_loaded {
            if let Some(victim) = self.order.pop_front() {
                self.engines.remove(&victim);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infer::device::DeviceRequest;
    use std::path::PathBuf;

    fn bundle_from_env() -> Option<PathBuf> {
        std::env::var_os("APOFASI_CHECKPOINT").map(PathBuf::from)
    }

    #[test]
    fn lru_evicts_least_recent() {
        let Some(bundle) = bundle_from_env() else {
            eprintln!("skip: set APOFASI_CHECKPOINT for registry LRU test");
            return;
        };
        // Only exercise eviction bookkeeping if multilingual exists.
        if CheckpointPaths::resolve_named(&bundle, CheckpointId::Multilingual).is_err() {
            eprintln!("skip: multilingual checkpoint missing under bundle");
            return;
        }
        let mut reg = CheckpointRegistry::open(&bundle, DeviceRequest::Cpu, 1).unwrap();
        let _ = reg.load(CheckpointId::English).unwrap();
        assert_eq!(reg.loaded(), vec![CheckpointId::English]);
        let _ = reg.load(CheckpointId::Multilingual).unwrap();
        assert_eq!(reg.loaded(), vec![CheckpointId::Multilingual]);
        assert!(!reg.loaded().contains(&CheckpointId::English));
    }
}
