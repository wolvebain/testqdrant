use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

// Integer

#[derive(Default, Debug, Deserialize, Serialize, JsonSchema, Clone, Copy, PartialEq, Hash, Eq)]
#[serde(rename_all = "snake_case")]
pub enum IntegerIndexType {
    #[default]
    Integer,
}

#[derive(Debug, Default, Deserialize, Serialize, JsonSchema, Clone, PartialEq, Hash, Eq)]
#[serde(rename_all = "snake_case")]
pub struct IntegerIndexParams {
    // Required for OpenAPI schema without anonymous types, versus #[serde(tag = "type")]
    pub r#type: IntegerIndexType,

    /// If true - support direct lookups.
    pub lookup: bool,

    /// If true - support ranges filters.
    pub range: bool,
}

// Text

#[derive(Default, Debug, Deserialize, Serialize, JsonSchema, Clone, Copy, PartialEq, Hash, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TextIndexType {
    #[default]
    Text,
}

#[derive(Default, Debug, Deserialize, Serialize, JsonSchema, Clone, Copy, PartialEq, Hash, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TokenizerType {
    Prefix,
    Whitespace,
    #[default]
    Word,
    Multilingual,
}

#[derive(Debug, Default, Deserialize, Serialize, JsonSchema, Clone, PartialEq, Hash, Eq)]
#[serde(rename_all = "snake_case")]
pub struct TextIndexParams {
    // Required for OpenAPI schema without anonymous types, versus #[serde(tag = "type")]
    pub r#type: TextIndexType,

    #[serde(default)]
    pub tokenizer: TokenizerType,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_token_len: Option<usize>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_token_len: Option<usize>,

    /// If true, lowercase all tokens. Default: true.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lowercase: Option<bool>,
}
