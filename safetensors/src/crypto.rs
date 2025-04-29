use crate::lib::HashMap;
use serde::{Deserialize, Serialize};

/// 加密信息结构体
#[derive(Debug, PartialEq, Eq, Clone, Serialize, Deserialize)]
pub struct EncInfo {
    /// 用于加密密钥的算法
    pub key_enc_algo: String,
    /// 用于加密数据的算法
    pub data_enc_algo: String,
    /// 密钥标识符
    pub key_id: String,
    /// 加密的密钥
    pub enc_key: Vec<u8>,
    /// 初始化向量
    pub iv: Vec<u8>,
    /// 认证标签
    pub tag: Vec<u8>,
}

/// 解析加密信息
pub fn parse_encryption_info(enc_str: &str) -> Result<HashMap<String, EncInfo>, serde_json::Error> {
    serde_json::from_str(enc_str)
}

/// 序列化加密信息
pub fn serialize_encryption_info(enc_info: &HashMap<String, EncInfo>) -> Result<String, serde_json::Error> {
    serde_json::to_string(enc_info)
} 