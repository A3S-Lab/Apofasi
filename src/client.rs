//! In-process System One client.

use crate::confidence::TemperatureTable;
use crate::engine::{DecisionEngine, LexicalEngine};
use crate::error::Result;
use crate::schema::{SystemOneRequest, SystemOneResponse};

#[cfg(feature = "router")]
use crate::route::RouteDecision;
#[cfg(feature = "router")]
use crate::router::Router;

/// High-level System One entry point.
#[derive(Debug, Clone)]
pub struct Client<E: DecisionEngine = LexicalEngine> {
    engine: E,
    #[cfg(feature = "router")]
    router: Router,
}

impl Default for Client<LexicalEngine> {
    fn default() -> Self {
        Self::lexical()
    }
}

impl Client<LexicalEngine> {
    /// Build a client with the size-minimal lexical engine.
    pub fn lexical() -> Self {
        Self {
            engine: LexicalEngine::default(),
            #[cfg(feature = "router")]
            router: Router::default(),
        }
    }

    /// Override softmax temperatures for the lexical engine.
    pub fn with_temperatures(mut self, temperatures: TemperatureTable) -> Self {
        self.engine.temperatures = temperatures;
        self
    }
}

impl<E: DecisionEngine> Client<E> {
    /// Wrap a custom engine.
    pub fn new(engine: E) -> Self {
        Self {
            engine,
            #[cfg(feature = "router")]
            router: Router::default(),
        }
    }

    /// Evaluate a System One request (Jev-compatible I/O).
    pub fn system_one(&self, request: SystemOneRequest) -> Result<SystemOneResponse> {
        self.engine.decide(&request)
    }

    /// Engine model id.
    pub fn model_id(&self) -> &str {
        self.engine.model_id()
    }

    /// Route without running the engine (`router` feature).
    #[cfg(feature = "router")]
    pub fn route(
        &self,
        request: &SystemOneRequest,
        model: Option<&str>,
        lang: Option<&str>,
    ) -> Result<RouteDecision> {
        self.router.route(request, model, lang)
    }

    /// Access the embedded router.
    #[cfg(feature = "router")]
    pub fn router(&self) -> &Router {
        &self.router
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitive::DecisionKind;
    use crate::schema::{Criteria, Question, State};
    use indexmap::IndexMap;
    use serde_json::json;

    #[test]
    fn system_one_returns_jev_shape() {
        let mut opts = IndexMap::new();
        opts.insert("billing".into(), Some(json!("refunds invoices")));
        opts.insert("other".into(), Some(json!("everything else")));
        let mut questions = IndexMap::new();
        questions.insert(
            "department".into(),
            Question::new(
                DecisionKind::Choice,
                json!("Which department?"),
                Some(Criteria::Choice(opts)),
            )
            .unwrap(),
        );
        let req = SystemOneRequest {
            model: Some("ignored-by-lexical".into()),
            state: State::Text("Please refund my invoice.".into()),
            questions,
        };
        let res = Client::default().system_one(req).unwrap();
        let v = serde_json::to_value(&res).unwrap();
        assert!(v["model"].as_str().unwrap().starts_with("apofasi-lexical-"));
        assert!(v["answers"]["department"]["choice"].is_string());
        assert!(v["answers"]["department"]["confidence"].is_number());
        assert!(v["usage"]["input_tokens"].as_u64().unwrap() > 0);
    }
}
