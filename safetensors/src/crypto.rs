use crate::tensor::{Metadata, TensorInfo};
use hex;
use once_cell::sync::OnceCell;
use ring::signature::{self, Ed25519KeyPair, UnparsedPublicKey};
use ring::{aead, rand::{self, SecureRandom}};
use serde::{Deserialize, Serialize, de::Error, Deserializer};
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use zeroize::Zeroizing;
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use std::path::Path;
use regorus::Engine;

/// Error types that can occur during cryptographic operations
#[derive(Debug)]
pub enum CryptoTensorError {
    /// Failed to unwrap (decrypt) a tensor's encryption key
    KeyUnwrap {
        /// The name of the tensor
        tensor: String,
        /// The source of the error
        source: String,
    },
    /// Failed to decrypt tensor data
    Decrypt {
        /// The name of the tensor
        tensor: String,
        /// The source of the error
        source: String,
    },
    /// Failed to access tensor data
    DataAccess {
        /// The name of the tensor
        tensor: String,
    },
    /// Invalid encryption algorithm specified
    InvalidAlgorithm(String),
    /// Invalid key length for the specified algorithm
    InvalidKeyLength {
        /// The expected length
        expected: usize,
        /// The actual length
        got: usize,
    },
    /// Invalid authentication tag length
    InvalidTagLength {
        /// The expected length
        expected: usize,
        /// The actual length
        got: usize,
    },
    /// Invalid initialization vector (IV) length
    InvalidIvLength {
        /// The expected length
        expected: usize,
        /// The actual length
        got: usize,
    },
    /// Failed to generate random data
    RandomGeneration(String),
    /// Failed to create encryption key
    KeyCreation(String),
    /// Failed to encrypt data
    Encryption(String),
    /// Failed to decrypt data
    Decryption(String),
    /// Master key is missing
    MissingMasterKey,
    /// Invalid key identifier
    InvalidKey(String),
    /// Failed to create signature
    Signing(String),
    /// Failed to verify signature
    Verification(String),
    /// Missing signing key
    MissingSigningKey,
    /// Missing verification key
    MissingVerificationKey,
    /// Missing signature in metadata
    MissingSignature(String),
    /// Invalid signature format
    InvalidSignatureFormat,
    /// Failed to load key from JWK
    KeyLoad {
        /// The source of the error
        source: String,
    },
    /// Invalid JWK URL format
    InvalidJwkUrl(String),
    /// No suitable key found in JWK
    NoSuitableKey,
    /// Multiple keys found without kid
    MultipleKeysWithoutKid,
    /// Policy相关错误
    Policy(String),
}

impl fmt::Display for CryptoTensorError {
    /// Format the error message for display
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CryptoTensorError::KeyUnwrap {
                tensor,
                source,
            } => {
                write!(
                    f,
                    "Failed to unwrap key for tensor {}: {}",
                    tensor, source
                )
            }
            CryptoTensorError::Decrypt {
                tensor,
                source,
            } => {
                write!(f, "Failed to decrypt tensor {}: {}", tensor, source)
            }
            CryptoTensorError::DataAccess { tensor } => {
                write!(f, "Failed to access data for tensor {}", tensor)
            }
            CryptoTensorError::InvalidAlgorithm(alg) => {
                write!(f, "Invalid algorithm: {}", alg)
            }
            CryptoTensorError::InvalidKeyLength { expected, got } => {
                write!(
                    f,
                    "Invalid key length: expected {} bytes, got {} bytes",
                    expected, got
                )
            }
            CryptoTensorError::InvalidTagLength { expected, got } => {
                write!(
                    f,
                    "Invalid tag length: expected {} bytes, got {} bytes",
                    expected, got
                )
            }
            CryptoTensorError::InvalidIvLength { expected, got } => {
                write!(
                    f,
                    "Invalid IV length: expected {} bytes, got {} bytes",
                    expected, got
                )
            }
            CryptoTensorError::RandomGeneration(e) => {
                write!(f, "Failed to generate random data: {}", e)
            }
            CryptoTensorError::KeyCreation(e) => {
                write!(f, "Failed to create key: {}", e)
            }
            CryptoTensorError::Encryption(e) => {
                write!(f, "Failed to encrypt data: {}", e)
            }
            CryptoTensorError::Decryption(e) => {
                write!(f, "Failed to decrypt data: {}", e)
            }
            CryptoTensorError::MissingMasterKey => {
                write!(f, "Master key is missing")
            }
            CryptoTensorError::InvalidKey(id) => {
                write!(f, "Invalid key identifier: {}", id)
            }
            CryptoTensorError::Signing(e) => {
                write!(f, "Failed to create signature: {}", e)
            }
            CryptoTensorError::Verification(e) => {
                write!(f, "Failed to verify signature: {}", e)
            }
            CryptoTensorError::MissingSigningKey => {
                write!(f, "Missing signing key")
            }
            CryptoTensorError::MissingVerificationKey => {
                write!(f, "Missing verification key")
            }
            CryptoTensorError::MissingSignature(msg) => {
                write!(f, "Missing signature: {}", msg)
            }
            CryptoTensorError::InvalidSignatureFormat => {
                write!(f, "Invalid signature format")
            }
            CryptoTensorError::KeyLoad { source } => {
                write!(f, "Failed to load key: {}", source)
            }
            CryptoTensorError::InvalidJwkUrl(url) => {
                write!(f, "Invalid JWK URL format: {}", url)
            }
            CryptoTensorError::NoSuitableKey => {
                write!(f, "No suitable key found in JWK")
            }
            CryptoTensorError::MultipleKeysWithoutKid => {
                write!(f, "Multiple keys found without kid")
            }
            CryptoTensorError::Policy(msg) => {
                write!(f, "Policy error: {}", msg)
            }
        }
    }
}

// TODO: Better error handling
impl std::error::Error for CryptoTensorError {}

impl From<crate::tensor::SafeTensorError> for CryptoTensorError {
    fn from(error: crate::tensor::SafeTensorError) -> Self {
        CryptoTensorError::Encryption(error.to_string())
    }
}

impl From<serde_json::Error> for CryptoTensorError {
    fn from(error: serde_json::Error) -> Self {
        CryptoTensorError::Encryption(error.to_string())
    }
}

impl From<String> for CryptoTensorError {
    fn from(error: String) -> Self {
        CryptoTensorError::Encryption(error)
    }
}

/// JWK URL
#[derive(Debug, Clone)]
struct JwkUrl {
    path: String,
}

impl JwkUrl {
    /// Parse a JWK URL
    fn parse(url: &str) -> Result<Self, CryptoTensorError> {
        if let Some((scheme, rest)) = url.split_once("://") {
            match scheme.to_lowercase().as_str() {
                "file" => (),
                _ => return Err(CryptoTensorError::InvalidJwkUrl(url.to_string())),
            };

            // Handle Unix path
            let path = if rest.starts_with('/') {
                rest.to_string()
            } else if rest.starts_with('~') {
                let home = std::env::var("HOME")
                    .map_err(|e| CryptoTensorError::KeyLoad { source: format!("Failed to get HOME: {}", e) })?;
                format!("{}{}", home, &rest[1..])
            } else {
                return Err(CryptoTensorError::InvalidJwkUrl("Only absolute paths and ~ are supported".to_string()));
            };

            Ok(Self { path })
        } else {
            Err(CryptoTensorError::InvalidJwkUrl("Missing URL scheme".to_string()))
        }
    }
}

/// Validation scenarios for Key Material
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ValidateMode {
    Save,
    Load,
    Jwk,
}

/// JSON Web Key (JWK) type
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum JwkKeyType {
    Oct,
    Okp,
}

impl JwkKeyType {
    /// Convert a string representation to a key type
    fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "oct" => Some(JwkKeyType::Oct),
            "okp" => Some(JwkKeyType::Okp),
            _ => None,
        }
    }
}

/// JSON Web Key (JWK) use
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum JwkKeyUse {
    Sig,
    Enc,
}

impl JwkKeyUse {
    /// Convert a string representation to a key use
    fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "sig" => Some(JwkKeyUse::Sig),
            "enc" => Some(JwkKeyUse::Enc),
            _ => None,
        }
    }
}

/// JSON Web Key (JWK) operations
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum JwkKeyOperation {
    Sign,
    Verify,
    Encrypt,
    Decrypt,
}

impl JwkKeyOperation {
    /// Convert a string representation to a key operation
    fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "sign" => Some(JwkKeyOperation::Sign),
            "verify" => Some(JwkKeyOperation::Verify),
            "encrypt" => Some(JwkKeyOperation::Encrypt),
            "decrypt" => Some(JwkKeyOperation::Decrypt),
            _ => None,
        }
    }
}

/// Serialize and deserialize OnceCell<Option<String>>
mod once_cell_option {
    use super::*;
    use once_cell::sync::OnceCell;

    pub fn deserialize<'de, D>(deserializer: D) -> Result<OnceCell<Option<String>>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value: Option<String> = Option::deserialize(deserializer)?;
        let cell = OnceCell::new();
        if let Some(v) = value {
            cell.set(Some(v)).map_err(|_| D::Error::custom("Failed to set OnceCell value"))?;
        }
        Ok(cell)
    }
}

/// Key Material structure for managing cryptographic keys
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyMaterial {
    #[serde(rename = "kty")]
    key_type: JwkKeyType,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "use")]
    use_: Option<JwkKeyUse>,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    key_ops: Option<Vec<JwkKeyOperation>>,
    
    alg: String,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    kid: Option<String>,
    
    /// The master key encoded in base64 for encryption
    #[serde(skip_serializing, default)]
    #[serde(with = "once_cell_option")]
    pub k: OnceCell<Option<String>>,

    /// The public key encoded in base64 for signing
    #[serde(skip_serializing, default)]
    #[serde(with = "once_cell_option")]
    #[serde(rename = "x")]
    pub x_pub: OnceCell<Option<String>>,

    /// The private key encoded in base64 for signing
    #[serde(skip_serializing, default)]
    #[serde(with = "once_cell_option")]
    #[serde(rename = "d")]
    pub d_priv: OnceCell<Option<String>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    jku: Option<String>,
}

impl KeyMaterial {
    /// Create a new KeyMaterial
    pub fn new(
        key_type: String,
        alg: String,
        use_: Option<String>,
        key_ops: Option<Vec<String>>,
        kid: Option<String>,
        jku: Option<String>,
        k: Option<Vec<u8>>,
        x_pub: Option<Vec<u8>>,
        d_priv: Option<Vec<u8>>,
    ) -> Result<Self, CryptoTensorError> {
        let key_material = Self {
            key_type: JwkKeyType::from_str(&key_type)
                .ok_or_else(|| CryptoTensorError::InvalidKey(format!("Invalid key type: {}", key_type)))?,
            use_: use_.as_deref()
                .map(|u| JwkKeyUse::from_str(u)
                    .ok_or_else(|| CryptoTensorError::InvalidKey(format!("Invalid key use: {}", u))))
                .transpose()?,
            key_ops: key_ops.map(|ops| {
                let valid_ops: Result<Vec<JwkKeyOperation>, CryptoTensorError> = ops.iter()
                    .map(|op| JwkKeyOperation::from_str(op)
                        .ok_or_else(|| CryptoTensorError::InvalidKey(format!("Invalid key operation: {}", op))))
                    .collect();
                valid_ops
            }).transpose()?,
            alg,
            kid,
            k: OnceCell::new(),
            x_pub: OnceCell::new(),
            d_priv: OnceCell::new(),
            jku,
        };

        if let Some(k) = k {
            key_material.k.set(Some(BASE64.encode(&k)))
                .map_err(|_| CryptoTensorError::KeyCreation("Failed to set key".to_string()))?;
        }
        if let Some(x_pub) = x_pub {
            key_material.x_pub.set(Some(BASE64.encode(&x_pub)))
                .map_err(|_| CryptoTensorError::KeyCreation("Failed to set public key".to_string()))?;
        }
        if let Some(d_priv) = d_priv {
            key_material.d_priv.set(Some(BASE64.encode(&d_priv)))
                .map_err(|_| CryptoTensorError::KeyCreation("Failed to set private key".to_string()))?;
        }

        Ok(key_material)
    }

    /// Load key from JWK
    pub fn load_key(&self) -> Result<(), CryptoTensorError> {
        // Try to load from jku first
        if let Some(jku) = &self.jku {
            if let Some(key) = self.try_load_key_from_path(jku)? {
                self.update_from_key(&key)?;
                return Ok(());
            }
        }

        // Try to load from environment variable
        if let Ok(path) = std::env::var("CRYPTOTENSOR_KEY_JKU") {
            if let Some(key) = self.try_load_key_from_path(&path)? {
                self.update_from_key(&key)?;
                return Ok(());
            }
        }

        // Try to load from default path
        let home = std::env::var("HOME").unwrap();
        let default_file = Path::new(&home).join(".cryptotensor/keys.jwk");
        if default_file.exists() {
            let default_path = format!("file://{}", default_file.display());
            if let Some(key) = self.try_load_key_from_path(&default_path)? {
                self.update_from_key(&key)?;
                return Ok(());
            }
        }

        Err(CryptoTensorError::NoSuitableKey)
    }

    /// Try to load key from a path
    fn try_load_key_from_path(&self, path: &str) -> Result<Option<KeyMaterial>, CryptoTensorError> {
        let url = JwkUrl::parse(path)?;
        let keys = self.load_keys_from_file(&url.path)?;
        if let Some(key) = self.select_key(&keys)? {
            return Ok(Some(key.clone()));
        }
        Ok(None)
    }

    /// Update this key material from another key material
    fn update_from_key(&self, key: &KeyMaterial) -> Result<(), CryptoTensorError> {
        // Update key fields based on key type
        match self.key_type {
            JwkKeyType::Oct => {
                if let Some(Some(k)) = key.k.get() {
                    self.k.set(Some(k.clone()))
                        .map_err(|_| CryptoTensorError::MissingMasterKey)?;
                } else {
                    return Err(CryptoTensorError::MissingMasterKey);
                }
            }
            JwkKeyType::Okp => {
                if let Some(Some(x_pub)) = key.x_pub.get() {
                    self.x_pub.set(Some(x_pub.clone()))
                        .map_err(|_| CryptoTensorError::MissingVerificationKey)?;
                } else {
                    return Err(CryptoTensorError::MissingVerificationKey);
                }
                if let Some(Some(d_priv)) = key.d_priv.get() {
                    self.d_priv.set(Some(d_priv.clone()))
                        .map_err(|_| CryptoTensorError::MissingSigningKey)?;
                } else {
                    return Err(CryptoTensorError::MissingSigningKey);
                }
            }
        }

        Ok(())
    }

    /// Validate the Key Material based on different scenarios
    fn validate(&self, mode: ValidateMode) -> Result<(), CryptoTensorError> {
        // Validate key type
        if self.key_type != JwkKeyType::Oct && self.key_type != JwkKeyType::Okp {
            return Err(CryptoTensorError::InvalidKey("Invalid key type".to_string()));
        }

        // Validate algorithm based on mode
        match mode {
            ValidateMode::Save | ValidateMode::Load => {
                if self.alg.is_empty() {
                    return Err(CryptoTensorError::InvalidAlgorithm("Missing alg field".to_string()));
                }
            }
            ValidateMode::Jwk => {
                // Algorithm is optional when loading JWK
            }
        }

        // Validate key existence based on mode and key type
        match mode {
            ValidateMode::Save => {
                match self.key_type {
                    JwkKeyType::Oct => {
                        if self.k.get().and_then(|k| k.as_ref()).is_none() {
                            return Err(CryptoTensorError::MissingMasterKey);
                        }
                    }
                    JwkKeyType::Okp => {
                        if self.d_priv.get().and_then(|k| k.as_ref()).is_none() {
                            return Err(CryptoTensorError::MissingSigningKey);
                        }
                    }
                }
            }
            ValidateMode::Jwk => {
                match self.key_type {
                    JwkKeyType::Oct => {
                        if self.k.get().and_then(|k| k.as_ref()).is_none() {
                            return Err(CryptoTensorError::MissingMasterKey);
                        }
                    }
                    JwkKeyType::Okp => {
                        if self.x_pub.get().and_then(|k| k.as_ref()).is_none() {
                            return Err(CryptoTensorError::MissingVerificationKey);
                        }
                    }
                }
            }
            ValidateMode::Load => {
                // Keys are not required during loading
            }
        }

        // Validate algorithm based on key type if algorithm is present
        if !self.alg.is_empty() {
            match self.key_type {
                JwkKeyType::Oct => {
                    if EncryptionAlgorithm::from_str(&self.alg).is_none() {
                        return Err(CryptoTensorError::InvalidAlgorithm("Invalid encryption algorithm".to_string()));
                    }
                }
                JwkKeyType::Okp => {
                    if SignatureAlgorithm::from_str(&self.alg).is_none() {
                        return Err(CryptoTensorError::InvalidAlgorithm("Invalid signature algorithm".to_string()));
                    }
                }
            }
        }

        // Validate consistency between use and operations if present
        if let Some(use_) = &self.use_ {
            if let Some(ops) = &self.key_ops {
                match use_ {
                    JwkKeyUse::Sig => {
                        if !ops.contains(&JwkKeyOperation::Sign) && !ops.contains(&JwkKeyOperation::Verify) {
                            return Err(CryptoTensorError::InvalidKey("Signature key use requires sign or verify operations".to_string()));
                        }
                    }
                    JwkKeyUse::Enc => {
                        if !ops.contains(&JwkKeyOperation::Encrypt) && !ops.contains(&JwkKeyOperation::Decrypt) {
                            return Err(CryptoTensorError::InvalidKey("Encryption key use requires encrypt or decrypt operations".to_string()));
                        }
                    }
                }
            }
        }
        
        Ok(())
    }

    /// Load keys from a file
    fn load_keys_from_file(&self, path: &str) -> Result<Vec<KeyMaterial>, CryptoTensorError> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| CryptoTensorError::KeyLoad { source: format!("Failed to read file {}: {}", path, e) })?;
        
        // Try to parse as a single key first
        if let Ok(key) = serde_json::from_str::<KeyMaterial>(&content) {
            key.validate(ValidateMode::Jwk)?;
            return Ok(vec![key]);
        }

        // Try to parse as a JWK set
        #[derive(Deserialize)]
        struct JwkSet {
            keys: Vec<KeyMaterial>,
        }
        let jwk_set: JwkSet = serde_json::from_str(&content)
            .map_err(|e| CryptoTensorError::KeyLoad { source: format!("Failed to parse JWK: {}", e) })?;
        
        // Validate each key
        for key in &jwk_set.keys {
            key.validate(ValidateMode::Jwk)?;
        }
        
        Ok(jwk_set.keys)
    }

    /// Select a key from a list of keys
    fn select_key<'a>(&self, keys: &'a [KeyMaterial]) -> Result<Option<&'a KeyMaterial>, CryptoTensorError> {
        // First filter by key type
        let matching_keys: Vec<_> = keys.iter()
            .filter(|k| k.key_type == self.key_type)
            .collect();

        if matching_keys.is_empty() {
            return Ok(None);
        }

        // Then filter by algorithm if the key has one
        let matching_keys: Vec<_> = matching_keys.iter()
            .filter(|k| k.alg.is_empty() || k.alg == self.alg)
            .collect();

        if matching_keys.is_empty() {
            return Ok(None);
        }

        // Finally, if we have a kid, try to find an exact match
        if let Some(kid) = &self.kid {
            if let Some(key) = matching_keys.iter().find(|k| k.kid.as_ref() == Some(kid)) {
                return Ok(Some(*key));
            }
            return Ok(None);
        }

        // If no kid is specified, return the first matching key
        match matching_keys.len() {
            1 => Ok(Some(matching_keys[0])),
            _ => Ok(Some(matching_keys[0]))
        }
    }
}


/// Configuration for serializing tensors with encryption
pub struct SerializeCryptoConfig {
    /// The names of the tensors to encrypt
    tensors: Option<Vec<String>>,
    /// The key material for encryption
    enc_key: KeyMaterial,
    /// The key material for signing
    sign_key: KeyMaterial,
    /// Policy for model loading and KMS validation
    policy: LoadPolicy,
}

impl SerializeCryptoConfig {
    /// Create a new configuration for serializing tensors with encryption
    pub fn new(
        tensors: Option<Vec<String>>,
        enc_key: KeyMaterial,
        sign_key: KeyMaterial,
        policy: LoadPolicy,
    ) -> Result<Self, CryptoTensorError> {
        enc_key.validate(ValidateMode::Save)?;
        sign_key.validate(ValidateMode::Save)?;
        Ok(Self { tensors, enc_key, sign_key, policy })
    }
}

/// Supported signature algorithms for header signing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SignatureAlgorithm {
    Ed25519,
}

impl SignatureAlgorithm {
    /// Convert a string representation to a signature algorithm
    fn from_str(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "ED25519" => Some(SignatureAlgorithm::Ed25519),
            _ => None,
        }
    }

    /// Convert the signature algorithm to its string representation
    fn to_string(&self) -> String {
        match self {
            SignatureAlgorithm::Ed25519 => "ED25519".to_string(),
        }
    }
}

/// Supported encryption algorithms for tensor data encryption
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EncryptionAlgorithm {
    Aes128Gcm,
    Aes256Gcm,
    ChaCha20Poly1305,
}

impl EncryptionAlgorithm {
    /// Convert a string representation to an encryption algorithm
    fn from_str(s: &str) -> Option<Self> {
        let normalized = s.replace('-', "").to_lowercase();
        match normalized.as_str() {
            "aes128gcm" => Some(EncryptionAlgorithm::Aes128Gcm),
            "aes256gcm" => Some(EncryptionAlgorithm::Aes256Gcm),
            "chacha20poly1305" => Some(EncryptionAlgorithm::ChaCha20Poly1305),
            _ => None,
        }
    }

    /// Convert the encryption algorithm to its string representation
    fn to_string(&self) -> String {
        match self {
            EncryptionAlgorithm::Aes128Gcm => "aes128gcm".to_string(),
            EncryptionAlgorithm::Aes256Gcm => "aes256gcm".to_string(),
            EncryptionAlgorithm::ChaCha20Poly1305 => "chacha20poly1305".to_string(),
        }
    }

    /// Get the appropriate AEAD algorithm from the ring crate
    fn get_aead_algo(&self) -> &'static aead::Algorithm {
        match self {
            EncryptionAlgorithm::Aes128Gcm => &aead::AES_128_GCM,
            EncryptionAlgorithm::Aes256Gcm => &aead::AES_256_GCM,
            EncryptionAlgorithm::ChaCha20Poly1305 => &aead::CHACHA20_POLY1305,
        }
    }

    /// Get the required key length in bytes for the algorithm
    fn key_len(&self) -> usize {
        match self {
            EncryptionAlgorithm::Aes128Gcm => 16,        // 128 bits
            EncryptionAlgorithm::Aes256Gcm => 32,        // 256 bits
            EncryptionAlgorithm::ChaCha20Poly1305 => 32, // 256 bits
        }
    }

    /// Get the authentication tag length in bytes for the algorithm
    fn tag_len(&self) -> usize {
        match self {
            EncryptionAlgorithm::Aes128Gcm => 16,
            EncryptionAlgorithm::Aes256Gcm => 16,
            EncryptionAlgorithm::ChaCha20Poly1305 => 16,
        }
    }

    /// Create an AEAD tag from raw bytes
    fn create_tag(&self, tag_bytes: &[u8]) -> Result<aead::Tag, String> {
        let expected_len = self.tag_len();
        if tag_bytes.len() != expected_len {
            return Err(format!(
                "Invalid tag length: expected {} bytes, got {} bytes",
                expected_len,
                tag_bytes.len()
            ));
        }

        let mut tag = [0u8; 16]; // All supported algorithms use 16-byte tags
        tag.copy_from_slice(tag_bytes);
        Ok(aead::Tag::from(tag))
    }
}

/// Encrypt data using the specified algorithm
///
/// This function performs in-place encryption of the input data using the specified
/// encryption algorithm. It generates a random nonce (IV) and returns it along with
/// the authentication tag.
///
/// # Arguments
///
/// * `in_out` - The buffer containing the data to encrypt. The encrypted data will be
///              written back to this buffer.
/// * `key` - The encryption key to use
/// * `algo_name` - The name of the encryption algorithm to use
///
/// # Returns
///
/// * `Ok((Vec<u8>, Vec<u8>))` - A tuple containing the nonce (IV) and authentication tag
/// * `Err(CryptoTensorError)` - If encryption fails
///
/// # Errors
///
/// * `InvalidKeyLength` - If the key length is invalid for the algorithm
/// * `InvalidAlgorithm` - If the algorithm name is not supported
/// * `RandomGeneration` - If random number generation fails
/// * `KeyCreation` - If key creation fails
/// * `Encryption` - If the encryption operation fails
fn encrypt_data(
    in_out: &mut [u8],
    key: &[u8],
    algo_name: &str,
) -> Result<(Vec<u8>, Vec<u8>), CryptoTensorError> {
    // If input is empty, return empty IV and tag
    if in_out.is_empty() {
    return Ok((Vec::new(), Vec::new()));
    }

    // Validate inputs
    let algo = EncryptionAlgorithm::from_str(algo_name)
    .ok_or_else(|| CryptoTensorError::InvalidAlgorithm(algo_name.to_string()))?;

    if key.is_empty() {
        return Err(CryptoTensorError::InvalidKeyLength {
            expected: algo.key_len(),
            got: 0,
        });
    }

    if key.len() != algo.key_len() {
        return Err(CryptoTensorError::InvalidKeyLength {
            expected: algo.key_len(),
            got: key.len(),
        });
    }

    // Create aead key
    let aead_algo = algo.get_aead_algo();
    let key = aead::UnboundKey::new(aead_algo, key)
        .map_err(|e| CryptoTensorError::KeyCreation(e.to_string()))?;
    let key = aead::LessSafeKey::new(key);

    // Generate a new nonce
    let mut nonce_bytes = vec![0u8; aead_algo.nonce_len()];
    let rng = rand::SystemRandom::new();
    rng.fill(&mut nonce_bytes)
        .map_err(|e| CryptoTensorError::RandomGeneration(e.to_string()))?;
    let nonce = aead::Nonce::assume_unique_for_key(nonce_bytes.clone().try_into().unwrap());

    // Encrypt the data in place
    let tag = key
        .seal_in_place_separate_tag(nonce, aead::Aad::empty(), in_out)
        .map_err(|e| CryptoTensorError::Encryption(e.to_string()))?;

    Ok((nonce_bytes, tag.as_ref().to_vec()))
}

/// Decrypt data using the specified algorithm
///
/// This function performs in-place decryption of the input data using the specified
/// encryption algorithm, nonce (IV), and authentication tag.
///
/// # Arguments
///
/// * `in_out` - The buffer containing the encrypted data. The decrypted data will be
///              written back to this buffer.
/// * `key` - The decryption key to use
/// * `algo_name` - The name of the encryption algorithm that was used
/// * `iv` - The nonce (IV) used during encryption
/// * `tag` - The authentication tag from encryption
///
/// # Returns
///
/// * `Ok(())` - If decryption succeeds
/// * `Err(CryptoTensorError)` - If decryption fails
///
/// # Errors
///
/// * `InvalidKeyLength` - If the key length is invalid for the algorithm
/// * `InvalidAlgorithm` - If the algorithm name is not supported
/// * `InvalidIvLength` - If the IV length is invalid
/// * `InvalidTagLength` - If the tag length is invalid
/// * `KeyCreation` - If key creation fails
/// * `Decryption` - If the decryption operation fails
fn decrypt_data(
    in_out: &mut [u8],
    key: &[u8],
    algo_name: &str,
    iv: &[u8],
    tag: &[u8],
) -> Result<(), CryptoTensorError> {
    // If all inputs are empty, this is an empty data case
    if in_out.is_empty() && iv.is_empty() && tag.is_empty() {
        return Ok(());
    }

    // Validate inputs
    let algo = EncryptionAlgorithm::from_str(algo_name)
    .ok_or_else(|| CryptoTensorError::InvalidAlgorithm(algo_name.to_string()))?;

    if key.is_empty() {
        return Err(CryptoTensorError::InvalidKeyLength {
            expected: algo.key_len(),
            got: 0,
        });
    }

    if key.len() != algo.key_len() {
        return Err(CryptoTensorError::InvalidKeyLength {
            expected: algo.key_len(),
            got: key.len(),
        });
    }

    let aead_algo = algo.get_aead_algo();
    if iv.is_empty() || tag.is_empty() {
        return Err(CryptoTensorError::InvalidIvLength {
            expected: aead_algo.nonce_len(),
            got: 0,
        });
    }
    
    let key = aead::UnboundKey::new(aead_algo, key)
        .map_err(|e| CryptoTensorError::KeyCreation(e.to_string()))?;
    let key = aead::LessSafeKey::new(key);

    let nonce =
        aead::Nonce::try_assume_unique_for_key(iv).map_err(|_e| CryptoTensorError::InvalidIvLength {
            expected: aead_algo.nonce_len(),
            got: iv.len(),
        })?;

    // Create tag using algorithm-specific method
    let tag = algo
        .create_tag(tag)
        .map_err(|_e| CryptoTensorError::InvalidTagLength {
            expected: algo.tag_len(),
            got: tag.len(),
        })?;

    // Decrypt in place using separate tag
    key.open_in_place_separate_tag(nonce, aead::Aad::empty(), tag, in_out, 0..)
        .map_err(|e| CryptoTensorError::Decryption(e.to_string()))?;

    Ok(())
}

/// Sign data using the specified algorithm
///
/// # Arguments
///
/// * `data` - The data to sign
/// * `key` - The signing key
/// * `algo_name` - The name of the signature algorithm to use
///
/// # Returns
///
/// * `Ok(Vec<u8>)` - The signature
/// * `Err(CryptoTensorError)` - If signing fails
fn sign_data(data: &[u8], key: &[u8], algo_name: &str) -> Result<Vec<u8>, CryptoTensorError> {
    // Validate inputs
    if key.is_empty() {
        return Err(CryptoTensorError::MissingSigningKey);
    }

    let algo = SignatureAlgorithm::from_str(algo_name)
        .ok_or_else(|| CryptoTensorError::InvalidAlgorithm(algo_name.to_string()))?;

    match algo {
        SignatureAlgorithm::Ed25519 => {
            // Create Ed25519 key pair from private key
            let key_pair = Ed25519KeyPair::from_pkcs8(key).map_err(|e| {
                CryptoTensorError::Signing(format!("Failed to create Ed25519 key pair: {}", e))
            })?;

            // Sign the data
            Ok(key_pair.sign(data).as_ref().to_vec())
        }
    }
}

/// Verify a signature using the specified algorithm
///
/// # Arguments
///
/// * `data` - The data that was signed
/// * `signature` - The signature to verify
/// * `key` - The verification key
/// * `algo_name` - The name of the signature algorithm that was used
///
/// # Returns
///
/// * `Ok(bool)` - True if the signature is valid, false otherwise
/// * `Err(CryptoTensorError)` - If verification fails
fn verify_signature(
    data: &[u8],
    signature: &[u8],
    key: &[u8],
    algo_name: &str,
) -> Result<bool, CryptoTensorError> {
    // Validate inputs
    if key.is_empty() {
        return Err(CryptoTensorError::MissingVerificationKey);
    }

    let algo = SignatureAlgorithm::from_str(algo_name)
        .ok_or_else(|| CryptoTensorError::InvalidAlgorithm(algo_name.to_string()))?;

    match algo {
        SignatureAlgorithm::Ed25519 => {
            // Create Ed25519 public key
            let public_key = UnparsedPublicKey::new(&signature::ED25519, key);

            // Verify the signature
            match public_key.verify(data, signature) {
                Ok(_) => Ok(true),
                Err(_) => Ok(false),
            }
        }
    }
}

/// Information about encrypted tensor data and methods for encryption/decryption
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorCryptor<'data> {
    /// Algorithm used for encryption
    #[serde(skip)]
    enc_algo: String,
    /// Encrypted tensor key
    wrapped_key: Vec<u8>,
    /// Initialization vector for key encryption
    key_iv: Vec<u8>,
    /// Authentication tag for key encryption
    key_tag: Vec<u8>,
    /// Initialization vector for data encryption
    iv: Vec<u8>,
    /// Authentication tag for data encryption
    tag: Vec<u8>,
    /// Buffer for decrypted data
    #[serde(skip)]
    buffer: OnceCell<Vec<u8>>,
    /// Master key for key encryption/decryption
    #[serde(skip)]
    master_key: Arc<[u8]>,
    /// Phantom data for lifetime tracking
    #[serde(skip)]
    _phantom: std::marker::PhantomData<&'data ()>,
}

impl<'data> TensorCryptor<'data> {
    /// Create a new TensorCryptor from key material
    ///
    /// # Arguments
    ///
    /// * `key_material` - The key material containing encryption key
    ///
    /// # Returns
    ///
    /// * `Ok(Self)` - If the key material contains valid encryption key
    /// * `Err(CryptoTensorError)` - If the key material is invalid or missing required key
    fn new(key_material: &KeyMaterial) -> Result<Self, CryptoTensorError> {
        if let Some(Some(k)) = key_material.k.get() {
            if let Some(alg) = EncryptionAlgorithm::from_str(&key_material.alg) {
                let decoded_key = BASE64.decode(k)
                    .map_err(|e| CryptoTensorError::KeyCreation(format!("Failed to decode base64 key: {}", e)))?;
                Ok(Self {
                    enc_algo: alg.to_string(),
                    wrapped_key: Vec::new(),
                    key_iv: Vec::new(),
                    key_tag: Vec::new(),
                    iv: Vec::new(),
                    tag: Vec::new(),
                    buffer: OnceCell::new(),
                    master_key: Arc::from(decoded_key),
                    _phantom: std::marker::PhantomData,
                })
            } else {
                Err(CryptoTensorError::InvalidAlgorithm(key_material.alg.clone()))
            }
        } else {
            Err(CryptoTensorError::MissingMasterKey)
        }
    }

    /// Generate a random key for data encryption
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<u8>)` - A randomly generated key of the appropriate length
    /// * `Err(CryptoTensorError)` - If key generation fails
    ///
    /// # Errors
    ///
    /// * `InvalidAlgorithm` - If the algorithm name is not supported
    /// * `RandomGeneration` - If random number generation fails
    fn random_key(&self) -> Result<Vec<u8>, CryptoTensorError> {
        let algo = EncryptionAlgorithm::from_str(&self.enc_algo)
            .ok_or_else(|| CryptoTensorError::InvalidAlgorithm(self.enc_algo.clone()))?;

        let mut key = vec![0u8; algo.key_len()];
        let rng = rand::SystemRandom::new();
        rng.fill(&mut key)
            .map_err(|e| CryptoTensorError::RandomGeneration(e.to_string()))?;
        Ok(key)
    }

    /// Wrap (encrypt) a key using the master key
    ///
    /// # Arguments
    ///
    /// * `key` - The key to wrap
    ///
    /// # Returns
    ///
    /// * `Ok(())` - If key wrapping succeeds
    /// * `Err(CryptoTensorError)` - If key wrapping fails
    ///
    /// # Errors
    ///
    /// * `Encryption` - If key encryption fails
    fn wrap_key(&mut self, key: &[u8]) -> Result<(), CryptoTensorError> {
        let mut key_buf = key.to_vec();
        let (key_iv, key_tag) =
            encrypt_data(&mut key_buf, &self.master_key, &self.enc_algo)?;
        self.wrapped_key = key_buf;
        self.key_iv = key_iv;
        self.key_tag = key_tag;
        Ok(())
    }

    /// Unwrap (decrypt) a key using the master key
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<u8>)` - The unwrapped key
    /// * `Err(CryptoTensorError)` - If key unwrapping fails
    ///
    /// # Errors
    ///
    /// * `MissingMasterKey` - If the master key is not set
    /// * `Decryption` - If key decryption fails
    fn unwrap_key(&self) -> Result<Vec<u8>, CryptoTensorError> {
        // Validate master key
        if self.master_key.is_empty() {
            return Err(CryptoTensorError::MissingMasterKey);
        }

        let mut data_key = self.wrapped_key.clone();
        decrypt_data(
            &mut data_key,
            &self.master_key,
            &self.enc_algo,
            &self.key_iv,
            &self.key_tag,
        )?;
        Ok(data_key)
    }

    /// Decrypt data using the master key
    ///
    /// # Arguments
    ///
    /// * `data` - The encrypted data to decrypt
    ///
    /// # Returns
    ///
    /// * `Ok(&[u8])` - A reference to the decrypted data
    /// * `Err(CryptoTensorError)` - If decryption fails
    ///
    /// # Errors
    ///
    /// * `KeyUnwrap` - If key unwrapping fails
    /// * `Decryption` - If data decryption fails
    pub fn decrypt(&'data self, data: &[u8]) -> Result<&'data [u8], CryptoTensorError> {
        self.buffer
            .get_or_try_init(|| {
                let data_key = Zeroizing::new(self.unwrap_key()?);

                let mut buffer = data.to_vec();
                decrypt_data(
                    &mut buffer,
                    data_key.as_slice(),
                    &self.enc_algo,
                    &self.iv,
                    &self.tag,
                )?;

                Ok(buffer)
            })
            .map(|vec_ref| vec_ref.as_slice())
    }

    /// Encrypt data using the master key
    ///
    /// # Arguments
    ///
    /// * `data` - The data to encrypt
    ///
    /// # Returns
    ///
    /// * `Ok(())` - If encryption succeeds
    /// * `Err(CryptoTensorError)` - If encryption fails
    ///
    /// # Errors
    ///
    /// * `RandomGeneration` - If key generation fails
    /// * `Encryption` - If data encryption fails
    /// * `KeyCreation` - If key creation fails
    fn encrypt(&mut self, data: &[u8]) -> Result<(), CryptoTensorError> {
        // Generate random data encryption key
        let data_key = Zeroizing::new(self.random_key()?);
        // Copy data to buffer, prepare in-place encryption
        let mut buffer = data.to_vec();
        let (iv, tag) = encrypt_data(&mut buffer, &data_key, &self.enc_algo)?;
        self.iv = iv;
        self.tag = tag;
        self.wrap_key(&data_key)?;
        self.buffer.set(buffer).ok();
        Ok(())
    }
}

/// Information about header signature and methods for signing/verifying
#[derive(Debug, Clone)]
struct HeaderSigner {
    /// The algorithm used for signing
    alg: String,
    /// Private key for signing
    priv_key: Option<Vec<u8>>,
    /// Public key for verification
    pub_key: Option<Vec<u8>>,
    /// The signature of the header
    signature: OnceCell<Vec<u8>>,
}

impl HeaderSigner {
    /// Create a new header signer from key material
    ///
    /// # Arguments
    ///
    /// * `key_material` - The key material containing signing keys
    ///
    /// # Returns
    ///
    /// * `Ok(Self)` - If the key material contains valid signing keys
    /// * `Err(CryptoTensorError)` - If the key material is invalid or missing required keys
    fn new(key_material: &KeyMaterial) -> Result<Self, CryptoTensorError> {
        let alg = SignatureAlgorithm::from_str(&key_material.alg)
            .ok_or_else(|| CryptoTensorError::InvalidAlgorithm(key_material.alg.clone()))?;

        let priv_key = key_material.d_priv.get()
            .and_then(|k| k.as_ref())
            .map(|k| BASE64.decode(k)
                .map_err(|e| CryptoTensorError::KeyCreation(format!("Failed to decode base64 private key: {}", e))))
            .transpose()?;

        let pub_key = key_material.x_pub.get()
            .and_then(|k| k.as_ref())
            .map(|k| BASE64.decode(k)
                .map_err(|e| CryptoTensorError::KeyCreation(format!("Failed to decode base64 public key: {}", e))))
            .transpose()?;

        Ok(Self {
            alg: alg.to_string(),
            priv_key,
            pub_key,
            signature: OnceCell::new(),
        })
    }

    /// Sign the header data
    fn sign(&self, data: &[u8]) -> Result<(), CryptoTensorError> {
        match &self.priv_key {
            Some(key) => {
                let signature = sign_data(data, key, &self.alg)?;
                self.signature
                    .set(signature)
                    .map_err(|_| CryptoTensorError::Signing("Signature already set".to_string()))?;
                Ok(())
            }
            None => Err(CryptoTensorError::MissingSigningKey),
        }
    }

    /// Verify the header signature
    fn verify(&self, data: &[u8]) -> Result<bool, CryptoTensorError> {
        match &self.pub_key {
            Some(key) => {
                match self.signature.get() {
                    Some(signature) => verify_signature(data, signature, key, &self.alg),
                    None => Err(CryptoTensorError::MissingSignature("No signature to verify".to_string())),
                }
            }
            None => Err(CryptoTensorError::MissingVerificationKey),
        }
    }
}

/// Policy for tensor model loading and remote KMS validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadPolicy {
    /// OPA policy content for tensor model loading validation
    #[serde(rename = "local")]
    local_policy: String,
    
    /// OPA policy content for KMS key release validation
    #[serde(rename = "remote")]
    remote_policy: String,
}

impl LoadPolicy {
    /// Create a new LoadPolicy
    pub fn new(local: Option<String>, remote: Option<String>) -> Self {
        let default_policy = "package model\nallow = true".to_string();
        Self {
            local_policy: local.unwrap_or_else(|| default_policy.clone()),
            remote_policy: remote.unwrap_or(default_policy),
        }
    }

    /// Validate tensor model loading using local Rego policy
    /// Currently does not validate based on input; input parameter is reserved for future use.
    pub fn evaluate(&self, _input: String) -> Result<bool, CryptoTensorError> {
        let mut engine = Engine::new();
        // Load local policy
        engine
            .add_policy(String::from("model.rego"), self.local_policy.clone())
            .map_err(|e| CryptoTensorError::Policy(format!("Failed to add policy: {e}")))?;

        // Input parsing and setting will be implemented in the future
        // let input_value = regorus::Value::from_json_str(&input)
        //     .map_err(|e| CryptoTensorError::Policy(format!("Invalid input JSON: {e}")))?;
        // engine.set_input(input_value);

        // Evaluate policy rule
        let result = engine
            .eval_rule(String::from("data.model.allow"))
            .map_err(|e| CryptoTensorError::Policy(format!("Policy evaluation failed: {e}")))?;

        // Parse result
        match result {
            regorus::Value::Bool(allowed) => Ok(allowed),
            regorus::Value::Undefined => Err(CryptoTensorError::Policy("Policy returned undefined".to_string())),
            _ => Err(CryptoTensorError::Policy("Policy did not return a boolean".to_string())),
        }
    }
}

/// Manager for handling encryption and decryption of multiple tensors
#[derive(Debug)]
pub struct CryptoTensor<'data> {
    /// Mapping from tensor names to their encryptors
    cryptors: HashMap<String, TensorCryptor<'data>>,
    /// Signer for signing/verifying the file header
    signer: HeaderSigner,
    /// Key material for encryption/decryption
    enc_key: KeyMaterial,
    /// Key material for signing/verification
    sign_key: KeyMaterial,
    /// Policy for model loading and KMS validation
    policy: LoadPolicy,
}

impl<'data> CryptoTensor<'data> {
    /// Get the encryptor for a specific tensor
    pub fn get(&self, tensor_name: &str) -> Option<&TensorCryptor<'data>> {
        self.cryptors.get(tensor_name)
    }

    /// Get a mutable reference to the encryptor for a specific tensor
    fn get_mut(&mut self, tensor_name: &str) -> Option<&mut TensorCryptor<'data>> {
        self.cryptors.get_mut(tensor_name)
    }

    /// Create a new encryptor mapping from encryption configuration
    ///
    /// # Arguments
    ///
    /// * `tensor_names` - List of all available tensor names
    /// * `config` - serialization configuration (it is up to the caller to ensure that the configuration exists)
    ///
    /// # Returns
    ///
    /// A new CryptoTensor instance. If no configuration is provided or no tensors
    /// are selected for encryption, the manager will be initialized without any
    /// encryptors.
    pub fn from_serialize_config(
        tensors: Vec<String>,
        config: &SerializeCryptoConfig,
    ) -> Result<Option<Self>, CryptoTensorError> {
        // Validate the key material
        config.enc_key.validate(ValidateMode::Save)?;
        config.sign_key.validate(ValidateMode::Save)?;

        // Determine which tensors need to be encrypted
        let matched_tensors = match &config.tensors {
            None => tensors,
            Some(names) => names
                .iter()
                .filter(|name| tensors.contains(name))
                .cloned()
                .collect(),
        };

        // Return None if no tensors need encryption
        if matched_tensors.is_empty() {
            return Ok(None);
        }

        // Create cryptor for each tensor
        let cryptors = matched_tensors
            .iter()
            .map(|name| {
                let cryptor = TensorCryptor::new(&config.enc_key)?;
                Ok((name.clone(), cryptor))
            })
            .collect::<Result<HashMap<String, TensorCryptor<'data>>, CryptoTensorError>>()?;

        // Create signer
        let signer = HeaderSigner::new(&config.sign_key)?;

        Ok(Some(Self {
            cryptors,
            enc_key: config.enc_key.clone(),
            sign_key: config.sign_key.clone(),
            signer,
            policy: config.policy.clone(),
        }))
    }

    /// Generate the metadata for serialization
    ///
    /// # Arguments
    ///
    /// * `tensors` - List of tensor names and their information
    /// * `metadata` - Optional metadata containing additional information
    ///
    /// # Returns
    ///
    /// * `Ok(Some(HashMap))` - If there are encryptors to serialize
    /// * `Ok(None)` - If there are no encryptors to serialize
    /// * `Err(CryptoTensorError)` - If serialization fails
    ///
    /// # Errors
    ///
    /// * `Encryption` - If JSON serialization fails
    pub fn generate_metadata(
        &self,
        tensors: Vec<(String, TensorInfo)>,
        metadata: Option<HashMap<String, String>>,
    ) -> Result<Option<HashMap<String, String>>, CryptoTensorError> {
        // Initialize with base metadata or empty HashMap
        let mut new_metadata = metadata.unwrap_or_default();

        // Add key material information
        let key_material = serde_json::json!({
            "enc": self.enc_key,
            "sign": self.sign_key
        });
        let key_material_json = serde_json::to_string(&key_material)
            .map_err(|e| CryptoTensorError::Encryption(e.to_string()))?;
        new_metadata.insert("__crypto_keys__".to_string(), key_material_json);

        // Add encryption information
        let crypto_json = serde_json::to_string(&self.cryptors)
            .map_err(|e| CryptoTensorError::Encryption(e.to_string()))?;
        new_metadata.insert("__encryption__".to_string(), crypto_json);

        // Add policy information
        let policy_json = serde_json::to_string(&self.policy)
            .map_err(|e| CryptoTensorError::Encryption(e.to_string()))?;
        new_metadata.insert("__policy__".to_string(), policy_json);

        // Add signature information
        let header = Metadata::new(Some(new_metadata.clone()), tensors)?;
        let header_json = serde_json::to_string(&header)?;
        self.signer.sign(&header_json.as_bytes())?;
        let signature = self.signer.signature.get()
            .ok_or_else(|| CryptoTensorError::Signing("Failed to get signature".to_string()))?;
        new_metadata.insert("__signature__".to_string(), hex::encode(signature));

        Ok(Some(new_metadata))
    }

    /// Create a new encryptor mapping from metadata
    ///
    /// # Arguments
    ///
    /// * `header` - The Metadata object containing the header
    ///
    /// # Returns
    ///
    /// * `Ok(CryptoTensor)` - If valid encryption metadata was found and verified
    /// * `Err(CryptoTensorError)` - If verification fails or metadata is invalid
    pub fn from_header(header: &Metadata) -> Result<Option<Self>, CryptoTensorError> {
        // return None if the header does not contain metadata or metadata does not contain encryption info
        let metadata = match header.metadata.as_ref() {
            Some(m) => m,
            None => return Ok(None),
        };
        let encryption_info = match metadata.get("__encryption__") {
            Some(info) => info,
            None => return Ok(None),
        };

        // Verify required fields exist
        let key_materials = metadata
            .get("__crypto_keys__")
            .ok_or_else(|| CryptoTensorError::InvalidKey("Missing __crypto_keys__ in metadata".to_string()))?;
        let signature_hex = metadata
            .get("__signature__")
            .ok_or_else(|| CryptoTensorError::MissingSignature("Missing __signature__ in metadata".to_string()))?;
        let policy_str = metadata.get("__policy__")
            .ok_or_else(|| CryptoTensorError::Policy("Missing __policy__ in metadata".to_string()))?;

        // Parse key materials
        let key_materials: serde_json::Value = serde_json::from_str(key_materials)
            .map_err(|e| CryptoTensorError::InvalidKey(format!("Failed to parse key materials: {}", e)))?;
        let enc_key: KeyMaterial = serde_json::from_value(key_materials["enc"].clone())
            .map_err(|e| CryptoTensorError::InvalidKey(format!("Failed to parse encryption key: {}", e)))?;
        let sign_key: KeyMaterial = serde_json::from_value(key_materials["sign"].clone())
            .map_err(|e| CryptoTensorError::InvalidKey(format!("Failed to parse signing key: {}", e)))?;
        enc_key.validate(ValidateMode::Load)?;
        sign_key.validate(ValidateMode::Load)?;

        // Load keys
        sign_key.load_key()?;

        // Create signer and verify signature
        let signer = HeaderSigner::new(&sign_key)?;
        let signature = hex::decode(signature_hex)
            .map_err(|_| CryptoTensorError::InvalidSignatureFormat)?;
        signer.signature.set(signature)
            .expect("Failed to set signature");
        let mut header_for_verify = header.clone();
        if let Some(metadata) = &mut header_for_verify.metadata {
            metadata.remove("__signature__");
        }
        let header_for_verify_json = serde_json::to_string(&header_for_verify)
            .map_err(|e| CryptoTensorError::Verification(e.to_string()))?;
        if !signer.verify(header_for_verify_json.as_bytes())? {
            return Err(CryptoTensorError::Verification("Signature verification failed".to_string()));
        }

        // Verify policy
        let policy: LoadPolicy = serde_json::from_str(policy_str)
            .map_err(|e| CryptoTensorError::Policy(format!("Failed to parse policy: {}", e)))?;
        if !policy.evaluate(String::new())? {
            return Err(CryptoTensorError::Policy("Policy evaluation denied".to_string()));
        }

        // Initialize cryptors after verification
        enc_key.load_key()?;
        let master_key: Arc<[u8]> = if let Some(Some(k)) = enc_key.k.get() {
            Arc::from(BASE64.decode(k)
                .map_err(|e| CryptoTensorError::KeyCreation(format!("Failed to decode base64 key: {}", e)))?)
        } else {
            return Err(CryptoTensorError::MissingMasterKey);
        };

        let mut cryptors: HashMap<String, TensorCryptor<'data>> = serde_json::from_str(encryption_info)
        .map_err(|e| CryptoTensorError::Encryption(e.to_string()))?;
        for cryptor in cryptors.values_mut() {
            cryptor.master_key = master_key.clone();
            cryptor.enc_algo = enc_key.alg.clone();
        }

        Ok(Some(Self {
            cryptors,
            signer,
            enc_key,
            sign_key,
            policy,
        }))
    }

    /// Silently decrypt data for a tensor
    ///
    /// If no encryptor exists for the tensor, returns the original data unchanged.
    ///
    /// # Arguments
    ///
    /// * `tensor_name` - The name of the tensor
    /// * `data` - The encrypted data to decrypt
    ///
    /// # Returns
    ///
    /// * `Ok(&[u8])` - The decrypted data, or the original data if no encryptor exists
    /// * `Err(CryptoTensorError)` - If decryption fails
    pub fn silent_decrypt(
        &'data self,
        tensor_name: &str,
        data: &'data [u8],
    ) -> Result<&'data [u8], CryptoTensorError> {
        match self.get(tensor_name) {
            Some(cryptor) => cryptor.decrypt(data),
            None => Ok(data), // Return original data if no cryptor is found
        }
    }

    /// Silently encrypt data for a tensor
    ///
    /// If no encryptor exists for the tensor, does nothing.
    ///
    /// # Arguments
    ///
    /// * `tensor_name` - The name of the tensor
    /// * `data` - The data to encrypt
    ///
    /// # Returns
    ///
    /// * `Ok(())` - If encryption succeeds or no encryptor exists
    /// * `Err(CryptoTensorError)` - If encryption fails
    ///
    /// # Errors
    ///
    /// * `RandomGeneration` - If key generation fails
    /// * `Encryption` - If data encryption fails
    /// * `KeyCreation` - If key creation fails
    pub fn silent_encrypt(
        &mut self,
        tensor_name: &str,
        data: &[u8],
    ) -> Result<(), CryptoTensorError> {
        match self.get_mut(tensor_name) {
            Some(cryptor) => cryptor.encrypt(data),
            None => Ok(()),
        }
    }

    /// Get encrypted data for a specific tensor
    ///
    /// # Arguments
    ///
    /// * `tensor_name` - The name of the tensor
    ///
    /// # Returns
    ///
    /// * `Some(&[u8])` - The encrypted data if available
    /// * `None` - If no encrypted data is available
    pub fn get_encrypted_data(&self, tensor_name: &str) -> Option<&[u8]> {
        match self.get(tensor_name) {
            Some(cryptor) => cryptor.buffer.get().map(|buf| buf.as_slice()),
            None => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ring::rand::SystemRandom;
    use ring::signature::{Ed25519KeyPair, KeyPair};
    use std::collections::HashMap;
    use std::path::Path;
    use std::fs;
    use tempfile;
    use crate::tensor::Dtype;

    /// Test encryption and decryption of empty tensor data
    #[test]
    fn test_tensor_cryptor_empty_data() -> Result<(), CryptoTensorError> {
        let master_key = vec![1u8; 32];
        let key_material = KeyMaterial::new(
            "oct".to_string(),
            "aes256gcm".to_string(),
            Some("enc".to_string()),
            Some(vec!["encrypt".to_string(), "decrypt".to_string()]),
            None,
            None,
            Some(master_key),
            None,
            None,
        )?;

        let mut encryptor = TensorCryptor::new(&key_material)?;
        let empty_data = b"";
        assert!(encryptor.encrypt(empty_data).is_ok());
        let encrypted_empty = encryptor.buffer.get().unwrap();
        assert!(encrypted_empty.is_empty()); // Should be empty encrypted data

        let mut decryptor = TensorCryptor::new(&key_material)?;
        decryptor.wrapped_key = encryptor.wrapped_key.clone();
        decryptor.key_iv = encryptor.key_iv.clone();
        decryptor.key_tag = encryptor.key_tag.clone();
        decryptor.iv = encryptor.iv.clone();
        decryptor.tag = encryptor.tag.clone();
        let decrypted_empty = decryptor.decrypt(encrypted_empty)?;
        assert_eq!(decrypted_empty, empty_data);
        Ok(())
    }

    /// Test encryption and decryption of non-empty tensor data
    #[test]
    fn test_tensor_cryptor_non_empty_data() -> Result<(), CryptoTensorError> {
        let master_key = vec![1u8; 32];
        let key_material = KeyMaterial::new(
            "oct".to_string(),
            "aes256gcm".to_string(),
            Some("enc".to_string()),
            Some(vec!["encrypt".to_string(), "decrypt".to_string()]),
            None,
            None,
            Some(master_key),
            None,
            None,
        )?;

        let mut encryptor = TensorCryptor::new(&key_material)?;
        let test_data = b"Hello";
        assert!(encryptor.encrypt(test_data).is_ok());
        let encrypted_data = encryptor.buffer.get().unwrap();
        assert_ne!(encrypted_data, test_data);

        let mut decryptor = TensorCryptor::new(&key_material)?;
        decryptor.wrapped_key = encryptor.wrapped_key.clone();
        decryptor.key_iv = encryptor.key_iv.clone();
        decryptor.key_tag = encryptor.key_tag.clone();
        decryptor.iv = encryptor.iv.clone();
        decryptor.tag = encryptor.tag.clone();
        let decrypted_data = decryptor.decrypt(encrypted_data)?;
        assert_eq!(decrypted_data, test_data);
        Ok(())
    }

    /// Test encryption and decryption of large tensor data
    #[test]
    fn test_tensor_cryptor_large_data() -> Result<(), CryptoTensorError> {
        let master_key = vec![1u8; 32];
        let key_material = KeyMaterial::new(
            "oct".to_string(),
            "aes256gcm".to_string(),
            Some("enc".to_string()),
            Some(vec!["encrypt".to_string(), "decrypt".to_string()]),
            None,
            None,
            Some(master_key),
            None,
            None,
        )?;

        let mut encryptor = TensorCryptor::new(&key_material)?;
        let large_data = vec![1u8; 1024];
        assert!(encryptor.encrypt(&large_data).is_ok());
        let encrypted_data = encryptor.buffer.get().unwrap();
        assert_ne!(encrypted_data, &large_data);

        let mut decryptor = TensorCryptor::new(&key_material)?;
        decryptor.wrapped_key = encryptor.wrapped_key.clone();
        decryptor.key_iv = encryptor.key_iv.clone();
        decryptor.key_tag = encryptor.key_tag.clone();
        decryptor.iv = encryptor.iv.clone();
        decryptor.tag = encryptor.tag.clone();
        let decrypted_data = decryptor.decrypt(encrypted_data)?;
        assert_eq!(decrypted_data, &large_data);
        Ok(())
    }

    /// Test encryption and decryption with all supported algorithms
    #[test]
    fn test_tensor_cryptor_all_algorithms() -> Result<(), CryptoTensorError> {
        let test_data = b"Test data for all algorithms";
        let algorithms = [
            ("aes128gcm", 16), // AES-128-GCM uses 16 bytes key
            ("aes256gcm", 32), // AES-256-GCM uses 32 bytes key
            ("chacha20poly1305", 32), // ChaCha20-Poly1305 uses 32 bytes key
        ];

        for (key_algo, key_len) in algorithms.iter() {
            let master_key = vec![1u8; *key_len];
            let key_material = KeyMaterial::new(
                "oct".to_string(),
                key_algo.to_string(),
                Some("enc".to_string()),
                Some(vec!["encrypt".to_string(), "decrypt".to_string()]),
                None,
                None,
                Some(master_key),
                None,
                None,
            )?;

            let mut encryptor = TensorCryptor::new(&key_material)?;
            assert!(encryptor.encrypt(test_data).is_ok());
            let encrypted_data = encryptor.buffer.get().unwrap();
            assert_ne!(encrypted_data, test_data);

            let mut decryptor = TensorCryptor::new(&key_material)?;
            decryptor.wrapped_key = encryptor.wrapped_key.clone();
            decryptor.key_iv = encryptor.key_iv.clone();
            decryptor.key_tag = encryptor.key_tag.clone();
            decryptor.iv = encryptor.iv.clone();
            decryptor.tag = encryptor.tag.clone();
            let decrypted_data = decryptor.decrypt(encrypted_data)?;
            assert_eq!(decrypted_data, test_data);
        }
        Ok(())
    }

    /// Test header signer signing and verification
    #[test]
    fn test_header_signer_sign_and_verify() -> Result<(), CryptoTensorError> {
        // Generate Ed25519 key pair
        let rng = SystemRandom::new();
        let pkcs8_bytes = Ed25519KeyPair::generate_pkcs8(&rng)
            .map_err(|e| CryptoTensorError::KeyCreation(e.to_string()))?;
        let key_pair = Ed25519KeyPair::from_pkcs8(pkcs8_bytes.as_ref())
            .map_err(|e| CryptoTensorError::KeyCreation(e.to_string()))?;
        let public_key = key_pair.public_key().as_ref().to_vec();

        // Create key material
        let key_material = KeyMaterial::new(
            "okp".to_string(),
            "ed25519".to_string(),
            Some("sig".to_string()),
            Some(vec!["sign".to_string(), "verify".to_string()]),
            None,
            None,
            None,
            Some(public_key),
            Some(pkcs8_bytes.as_ref().to_vec()),
        )?;

        // Create HeaderSigner for signing
        let signer = HeaderSigner::new(&key_material)?;

        // Test data
        let test_data = b"Test data for signing";

        // Sign data
        signer.sign(test_data)?;
        let signature = signer.signature.get()
            .ok_or_else(|| CryptoTensorError::Signing("Failed to get signature".to_string()))?
            .clone();

        // Create HeaderSigner for verification
        let verifier = HeaderSigner::new(&key_material)?;
        verifier.signature.set(signature)
            .map_err(|_| CryptoTensorError::Signing("Failed to set signature".to_string()))?;

        // Verify signature
        let is_valid = verifier.verify(test_data)?;
        assert!(is_valid);

        // Verify modified data
        let modified_data = b"Modified test data";
        let is_valid = verifier.verify(modified_data)?;
        assert!(!is_valid);

        Ok(())
    }

    /// Test header signer error handling
    #[test]
    fn test_header_signer_error_handling() -> Result<(), CryptoTensorError> {
        // Test missing private key
        let key_material = KeyMaterial::new(
            "okp".to_string(),
            "ed25519".to_string(),
            Some("sig".to_string()),
            Some(vec!["sign".to_string(), "verify".to_string()]),
            None,
            None,
            None,
            Some(vec![0u8; 32]), // Invalid public key
            None, // No private key
        )?;

        let signer = HeaderSigner::new(&key_material)?;
        let test_data = b"Test data";
        
        // Attempt to sign should fail
        assert!(matches!(
            signer.sign(test_data),
            Err(CryptoTensorError::MissingSigningKey)
        ));

        // Test missing public key
        let key_material = KeyMaterial::new(
            "okp".to_string(),
            "ed25519".to_string(),
            Some("sig".to_string()),
            Some(vec!["sign".to_string(), "verify".to_string()]),
            None,
            None,
            None,
            None, // No public key
            Some(vec![0u8; 32]), // Invalid private key
        )?;

        let signer = HeaderSigner::new(&key_material)?;
        
        // Attempt to verify should fail
        assert!(matches!(
            signer.verify(test_data),
            Err(CryptoTensorError::MissingVerificationKey)
        ));

        // Test invalid signature algorithm
        let key_material = KeyMaterial::new(
            "okp".to_string(),
            "invalid_algorithm".to_string(), // Invalid algorithm
            Some("sig".to_string()),
            Some(vec!["sign".to_string(), "verify".to_string()]),
            None,
            None,
            None,
            Some(vec![0u8; 32]),
            Some(vec![0u8; 32]),
        )?;

        // Creating HeaderSigner should fail
        assert!(matches!(
            HeaderSigner::new(&key_material),
            Err(CryptoTensorError::InvalidAlgorithm(_))
        ));

        Ok(())
    }

    /// Test if two repetitive signing will generate the same signature
    #[test]
    fn test_header_signer_signature_repeat() -> Result<(), CryptoTensorError> {
        // Generate Ed25519 key pair
        let rng = SystemRandom::new();
        let pkcs8_bytes = Ed25519KeyPair::generate_pkcs8(&rng)
            .map_err(|e| CryptoTensorError::KeyCreation(e.to_string()))?;
        let key_pair = Ed25519KeyPair::from_pkcs8(pkcs8_bytes.as_ref())
            .map_err(|e| CryptoTensorError::KeyCreation(e.to_string()))?;
        let public_key = key_pair.public_key().as_ref().to_vec();

        // Create key material
        let key_material = KeyMaterial::new(
            "okp".to_string(),
            "ed25519".to_string(),
            Some("sig".to_string()),
            Some(vec!["sign".to_string(), "verify".to_string()]),
            None,
            None,
            None,
            Some(public_key),
            Some(pkcs8_bytes.as_ref().to_vec()),
        )?;

        // Test data
        let test_data = b"Test data";

        // First signing
        let signer1 = HeaderSigner::new(&key_material)?;
        signer1.sign(test_data)?;
        let first_signature = signer1.signature.get().unwrap().clone();

        // Second signing (using new HeaderSigner instance)
        let signer2 = HeaderSigner::new(&key_material)?;
        signer2.sign(test_data)?;
        let second_signature = signer2.signature.get().unwrap().clone();

        // Verify both signatures are identical
        assert_eq!(first_signature, second_signature);

        Ok(())
    }

    /// Test KeyMaterial creation and validation
    #[test]
    fn test_key_material_creation_and_validation() {
        // Test normal encryption key creation
        let enc_key = KeyMaterial::new(
            "oct".to_string(),
            "aes256gcm".to_string(),
            Some("enc".to_string()),
            Some(vec!["encrypt".to_string(), "decrypt".to_string()]),
            Some("test-enc-key".to_string()),
            Some("file:///test/enc.jwk".to_string()),
            Some(vec![1u8; 32]),
            None,
            None,
        ).unwrap();

        assert_eq!(enc_key.key_type, JwkKeyType::Oct);
        assert_eq!(enc_key.alg, "aes256gcm");
        assert_eq!(enc_key.use_, Some(JwkKeyUse::Enc));

        // Test normal signing key creation
        let sign_key = KeyMaterial::new(
            "okp".to_string(),
            "ed25519".to_string(),
            Some("sig".to_string()),
            Some(vec!["sign".to_string(), "verify".to_string()]),
            Some("test-sign-key".to_string()),
            Some("file:///test/sign.jwk".to_string()),
            None,
            Some(vec![1u8; 32]),
            Some(vec![1u8; 32]),
        ).unwrap();

        assert_eq!(sign_key.key_type, JwkKeyType::Okp);
        assert_eq!(sign_key.alg, "ed25519");
        assert_eq!(sign_key.use_, Some(JwkKeyUse::Sig));

        // Test error cases
        // 1. Invalid key type
        let result = KeyMaterial::new(
            "invalid_type".to_string(),
            "aes256gcm".to_string(),
            Some("enc".to_string()),
            Some(vec!["encrypt".to_string(), "decrypt".to_string()]),
            Some("test-key".to_string()),
            Some("file:///test.jwk".to_string()),
            Some(vec![1u8; 32]),
            None,
            None,
        );
        assert!(matches!(result, Err(CryptoTensorError::InvalidKey(_))));

        // 2. Invalid key use
        let result = KeyMaterial::new(
            "oct".to_string(),
            "aes256gcm".to_string(),
            Some("invalid_use".to_string()),
            Some(vec!["encrypt".to_string(), "decrypt".to_string()]),
            Some("test-key".to_string()),
            Some("file:///test.jwk".to_string()),
            Some(vec![1u8; 32]),
            None,
            None,
        );
        assert!(matches!(result, Err(CryptoTensorError::InvalidKey(_))));

        // 3. Invalid operation type
        let result = KeyMaterial::new(
            "oct".to_string(),
            "aes256gcm".to_string(),
            Some("enc".to_string()),
            Some(vec!["invalid_op".to_string()]),
            Some("test-key".to_string()),
            Some("file:///test.jwk".to_string()),
            Some(vec![1u8; 32]),
            None,
            None,
        );
        assert!(matches!(result, Err(CryptoTensorError::InvalidKey(_))));

        // 4. Test validation in Save mode
        // 4.1 Test missing alg in Save mode (should error)
        let save_key = KeyMaterial::new(
            "oct".to_string(),
            "".to_string(), // Missing algorithm
            Some("enc".to_string()),
            Some(vec!["encrypt".to_string(), "decrypt".to_string()]),
            Some("test-key".to_string()),
            Some("file:///test.jwk".to_string()),
            Some(vec![1u8; 32]),
            None,
            None,
        ).unwrap();
        assert!(matches!(
            save_key.validate(ValidateMode::Save),
            Err(CryptoTensorError::InvalidAlgorithm(_))
        ));

        // 4.2 Test missing jku in Save mode (should pass)
        let save_key = KeyMaterial::new(
            "oct".to_string(),
            "aes256gcm".to_string(),
            Some("enc".to_string()),
            Some(vec!["encrypt".to_string(), "decrypt".to_string()]),
            Some("test-key".to_string()),
            None, // Missing jku
            Some(vec![1u8; 32]),
            None,
            None,
        ).unwrap();
        assert!(save_key.validate(ValidateMode::Save).is_ok());

        // 4.3 Test missing key in Save mode (should error)
        let save_key = KeyMaterial::new(
            "oct".to_string(),
            "aes256gcm".to_string(),
            Some("enc".to_string()),
            Some(vec!["encrypt".to_string(), "decrypt".to_string()]),
            Some("test-key".to_string()),
            Some("file:///test.jwk".to_string()),
            None, // Missing key
            None,
            None,
        ).unwrap();
        assert!(matches!(
            save_key.validate(ValidateMode::Save),
            Err(CryptoTensorError::MissingMasterKey)
        ));

        // 4.4 Test invalid algorithm in Save mode (should error)
        let save_key = KeyMaterial::new(
            "oct".to_string(),
            "invalid_algorithm".to_string(), // Invalid algorithm
            Some("enc".to_string()),
            Some(vec!["encrypt".to_string(), "decrypt".to_string()]),
            Some("test-key".to_string()),
            Some("file:///test.jwk".to_string()),
            Some(vec![1u8; 32]),
            None,
            None,
        ).unwrap();
        assert!(matches!(
            save_key.validate(ValidateMode::Save),
            Err(CryptoTensorError::InvalidAlgorithm(_))
        ));

        // 5. Test validation in Load mode
        // 5.1 Test missing jku in Load mode (should pass)
        let load_key = KeyMaterial::new(
            "oct".to_string(),
            "aes256gcm".to_string(),
            Some("enc".to_string()),
            Some(vec!["encrypt".to_string(), "decrypt".to_string()]),
            Some("test-key".to_string()),
            None, // Missing jku
            Some(vec![1u8; 32]),
            None,
            None,
        ).unwrap();
        assert!(load_key.validate(ValidateMode::Load).is_ok());

        // 5.2 Test missing key in Load mode (should pass)
        let load_key = KeyMaterial::new(
            "oct".to_string(),
            "aes256gcm".to_string(),
            Some("enc".to_string()),
            Some(vec!["encrypt".to_string(), "decrypt".to_string()]),
            Some("test-key".to_string()),
            Some("file:///test.jwk".to_string()),
            None, // Missing key
            None,
            None,
        ).unwrap();
        assert!(load_key.validate(ValidateMode::Load).is_ok());

        // 5.3 Test missing algorithm in Load mode (should error)
        let load_key = KeyMaterial::new(
            "oct".to_string(),
            "".to_string(), // Missing algorithm
            Some("enc".to_string()),
            Some(vec!["encrypt".to_string(), "decrypt".to_string()]),
            Some("test-key".to_string()),
            Some("file:///test.jwk".to_string()),
            Some(vec![1u8; 32]),
            None,
            None,
        ).unwrap();
        assert!(matches!(
            load_key.validate(ValidateMode::Load),
            Err(CryptoTensorError::InvalidAlgorithm(_))
        ));

        // 5.4 Test invalid algorithm in Load mode (should error)
        let load_key = KeyMaterial::new(
            "oct".to_string(),
            "invalid_algorithm".to_string(), // Invalid algorithm
            Some("enc".to_string()),
            Some(vec!["encrypt".to_string(), "decrypt".to_string()]),
            Some("test-key".to_string()),
            Some("file:///test.jwk".to_string()),
            Some(vec![1u8; 32]),
            None,
            None,
        ).unwrap();
        assert!(matches!(
            load_key.validate(ValidateMode::Load),
            Err(CryptoTensorError::InvalidAlgorithm(_))
        ));

        // 6. Test key validation in Save mode
        // 6.1 Test enc type without providing key (should error)
        let enc_key = KeyMaterial::new(
            "oct".to_string(),
            "aes256gcm".to_string(),
            Some("enc".to_string()),
            Some(vec!["encrypt".to_string(), "decrypt".to_string()]),
            Some("test-key".to_string()),
            Some("file:///test.jwk".to_string()),
            None, // No key provided
            None,
            None,
        ).unwrap();
        assert!(matches!(
            enc_key.validate(ValidateMode::Save),
            Err(CryptoTensorError::MissingMasterKey)
        ));

        // 6.2 Test sig type without providing private key (should error)
        let sign_key = KeyMaterial::new(
            "okp".to_string(),
            "ed25519".to_string(),
            Some("sig".to_string()),
            Some(vec!["sign".to_string(), "verify".to_string()]),
            Some("test-key".to_string()),
            Some("file:///test.jwk".to_string()),
            None,
            Some(vec![1u8; 32]), // Only public key
            None, // No private key
        ).unwrap();
        assert!(matches!(
            sign_key.validate(ValidateMode::Save),
            Err(CryptoTensorError::MissingSigningKey)
        ));

        // 6.3 Test sig type without providing public key (should pass)
        let sign_key = KeyMaterial::new(
            "okp".to_string(),
            "ed25519".to_string(),
            Some("sig".to_string()),
            Some(vec!["sign".to_string(), "verify".to_string()]),
            Some("test-key".to_string()),
            Some("file:///test.jwk".to_string()),
            None,
            None, // No public key
            Some(vec![1u8; 32]), // Only private key
        ).unwrap();
        assert!(sign_key.validate(ValidateMode::Save).is_ok());
    }

    /// Test KeyMaterial key loading functionality
    #[test]
    fn test_key_material_load_single_key_from_jku() -> Result<(), CryptoTensorError> {
        // Create temporary directory and test file
        let temp_dir = tempfile::tempdir().unwrap();
        let test_file = temp_dir.path().join("test.jwk");

        // Write single key JWK file content directly
        let jwk_content = r#"{
            "kty": "oct",
            "use": "enc",
            "key_ops": ["encrypt", "decrypt"],
            "alg": "aes256gcm",
            "kid": "test-key",
            "k": "AQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQE="
        }"#;

        // Write to file
        fs::write(&test_file, jwk_content)
            .map_err(|e| CryptoTensorError::KeyLoad { source: e.to_string() })?;

        // Create KeyMaterial and attempt to load key
        let key_material = KeyMaterial::new(
            "oct".to_string(),
            "aes256gcm".to_string(),
            Some("enc".to_string()),
            Some(vec!["encrypt".to_string(), "decrypt".to_string()]),
            Some("test-key".to_string()),
            Some(format!("file://{}", test_file.to_str().unwrap())),
            None,
            None,
            None,
        )?;

        // Load key
        assert!(key_material.load_key().is_ok());
        assert!(key_material.k.get().is_some());
        assert!(key_material.validate(ValidateMode::Load).is_ok());

        Ok(())
    }

    /// Test KeyMaterial multiple keys loading functionality
    #[test]
    fn test_key_material_load_multi_keys_from_jku() -> Result<(), CryptoTensorError> {
        // Create temporary directory and test file
        let temp_dir = tempfile::tempdir().unwrap();
        let test_file = temp_dir.path().join("test.jwk");

        // Write JWK Set file content
        let jwk_set_content = r#"{
            "keys": [
                {
                    "kty": "oct",
                    "use": "enc",
                    "key_ops": ["encrypt", "decrypt"],
                    "alg": "aes256gcm",
                    "kid": "key1",
                    "k": "AQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQE="
                },
                {
                    "kty": "oct",
                    "use": "enc",
                    "key_ops": ["encrypt", "decrypt"],
                    "alg": "aes256gcm",
                    "kid": "key2",
                    "k": "AgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAg="
                }
            ]
        }"#;

        // Write to file
        fs::write(&test_file, jwk_set_content)
            .map_err(|e| CryptoTensorError::KeyLoad { source: e.to_string() })?;

        // Test loading first key
        let key_material = KeyMaterial::new(
            "oct".to_string(),
            "aes256gcm".to_string(),
            Some("enc".to_string()),
            Some(vec!["encrypt".to_string(), "decrypt".to_string()]),
            Some("key1".to_string()),
            Some(format!("file://{}", test_file.to_str().unwrap())),
            None,
            None,
            None,
        )?;

        // Load key
        let load_result = key_material.load_key();
        assert!(load_result.is_ok(), "Failed to load encryption key: {}", load_result.unwrap_err());
        assert!(key_material.k.get().is_some());

        // Test loading second key
        let key_material = KeyMaterial::new(
            "oct".to_string(),
            "aes256gcm".to_string(),
            Some("enc".to_string()),
            Some(vec!["encrypt".to_string(), "decrypt".to_string()]),
            Some("key2".to_string()),
            Some(format!("file://{}", test_file.to_str().unwrap())),
            None,
            None,
            None,
        )?;

        // Load key
        let load_result = key_material.load_key();
        assert!(load_result.is_ok(), "Failed to load encryption key: {}", load_result.unwrap_err());
        assert!(key_material.k.get().is_some());

        // Test loading non-existent key
        let key_material = KeyMaterial::new(
            "oct".to_string(),
            "aes256gcm".to_string(),
            Some("enc".to_string()),
            Some(vec!["encrypt".to_string(), "decrypt".to_string()]),
            Some("non-existent".to_string()),
            Some(format!("file://{}", test_file.to_str().unwrap())),
            None,
            None,
            None,
        )?;

        // Loading key should fail
        let load_result = key_material.load_key();
        assert!(matches!(
            load_result,
            Err(CryptoTensorError::NoSuitableKey)
        ));

        Ok(())
    }

    /// Test loading key from environment variable
    #[test]
    fn test_key_material_load_multi_keys_from_env() -> Result<(), CryptoTensorError> {
        // Create temporary directory and test file
        let temp_dir = tempfile::tempdir().unwrap();
        let test_file = temp_dir.path().join("test.jwk");

        // Write JWK Set file content
        let jwk_set_content = r#"{
            "keys": [
                {
                    "kty": "oct",
                    "use": "enc",
                    "key_ops": ["encrypt", "decrypt"],
                    "alg": "aes256gcm",
                    "kid": "key1",
                    "k": "AQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQE="
                },
                {
                    "kty": "okp",
                    "use": "sig",
                    "key_ops": ["sign", "verify"],
                    "alg": "ed25519",
                    "kid": "key2",
                    "x": "AQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQE=",
                    "d": "AgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAg="
                }
            ]
        }"#;

        // Write to file
        fs::write(&test_file, jwk_set_content)
            .map_err(|e| CryptoTensorError::KeyLoad { source: e.to_string() })?;

        // Set environment variable
        let file_path = format!("file://{}", test_file.to_str().unwrap());
        std::env::set_var("CRYPTOTENSOR_KEY_JKU", file_path);

        // Test loading encryption key
        let enc_key = KeyMaterial::new(
            "oct".to_string(),
            "aes256gcm".to_string(),
            Some("enc".to_string()),
            Some(vec!["encrypt".to_string(), "decrypt".to_string()]),
            Some("key1".to_string()),
            None,
            None,
            None,
            None,
        )?;

        // Load key
        let load_result = enc_key.load_key();
        assert!(load_result.is_ok(), "Failed to load encryption key: {}", load_result.unwrap_err());
        assert!(enc_key.k.get().is_some());
        assert!(enc_key.validate(ValidateMode::Load).is_ok());

        // Test loading signing key
        let sign_key = KeyMaterial::new(
            "okp".to_string(),
            "ed25519".to_string(),
            Some("sig".to_string()),
            Some(vec!["sign".to_string(), "verify".to_string()]),
            Some("key2".to_string()),
            None,
            None,
            None,
            None,
        )?;

        // Load key
        let load_result = sign_key.load_key();
        assert!(load_result.is_ok(), "Failed to load signing key: {}", load_result.unwrap_err());
        assert!(sign_key.x_pub.get().is_some());
        assert!(sign_key.validate(ValidateMode::Load).is_ok());

        // Clean up environment variable
        std::env::remove_var("CRYPTOTENSOR_KEY_JKU");

        Ok(())
    }

    /// Test loading key from default path
    #[test]
    fn test_key_material_load_multi_keys_from_default_path() -> Result<(), CryptoTensorError> {
        // Create default directory
        let home = std::env::var("HOME").unwrap();
        let default_dir = Path::new(&home).join(".cryptotensor");
        fs::create_dir_all(&default_dir).unwrap();
        let default_file = default_dir.join("keys.jwk");

        // Write JWK Set file content
        let jwk_set_content = r#"{
            "keys": [
                {
                    "kty": "oct",
                    "use": "enc",
                    "key_ops": ["encrypt", "decrypt"],
                    "alg": "aes256gcm",
                    "kid": "key1",
                    "k": "AQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQE="
                },
                {
                    "kty": "okp",
                    "use": "sig",
                    "key_ops": ["sign", "verify"],
                    "alg": "ed25519",
                    "kid": "key2",
                    "x": "AQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQE=",
                    "d": "AgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAg="
                }
            ]
        }"#;

        // Write to file
        fs::write(&default_file, jwk_set_content)
            .map_err(|e| CryptoTensorError::KeyLoad { source: e.to_string() })?;

        // Test loading encryption key
        let enc_key = KeyMaterial::new(
            "oct".to_string(),
            "aes256gcm".to_string(),
            Some("enc".to_string()),
            Some(vec!["encrypt".to_string(), "decrypt".to_string()]),
            Some("key1".to_string()),
            None,
            None,
            None,
            None,
        )?;

        // Load key
        let load_result = enc_key.load_key();
        assert!(load_result.is_ok(), "Failed to load encryption key: {}", load_result.unwrap_err());
        assert!(enc_key.k.get().is_some());
        assert!(enc_key.validate(ValidateMode::Load).is_ok());

        // Test loading signing key
        let sign_key = KeyMaterial::new(
            "okp".to_string(),
            "ed25519".to_string(),
            Some("sig".to_string()),
            Some(vec!["sign".to_string(), "verify".to_string()]),
            Some("key2".to_string()),
            None,
            None,
            None,
            None,
        )?;

        // Load key
        let load_result = sign_key.load_key();
        assert!(load_result.is_ok(), "Failed to load signing key: {}", load_result.unwrap_err());
        assert!(sign_key.x_pub.get().is_some());
        assert!(sign_key.validate(ValidateMode::Load).is_ok());

        // Clean up test files
        fs::remove_file(&default_file).unwrap();
        fs::remove_dir(&default_dir).unwrap();

        Ok(())
    }

    /// Test the complete flow of CryptoTensor
    #[test]
    fn test_crypto_tensor_complete_flow() -> Result<(), CryptoTensorError> {
        // Create temporary directory for test files
        let temp_dir = tempfile::tempdir().unwrap();
        let test_file = temp_dir.path().join("test.jwk");

        // Create test key material for encryption
        let enc_key = KeyMaterial::new(
            "oct".to_string(),
            "aes256gcm".to_string(),
            Some("enc".to_string()),
            Some(vec!["encrypt".to_string(), "decrypt".to_string()]),
            Some("test-enc-key".to_string()),
            Some(format!("file://{}", test_file.to_str().unwrap())),
            Some(vec![1u8; 32]),
            None,
            None,
        )?;

        // Generate Ed25519 key pair for signing
        let rng = SystemRandom::new();
        let pkcs8_bytes = Ed25519KeyPair::generate_pkcs8(&rng)
            .map_err(|e| CryptoTensorError::KeyCreation(e.to_string()))?;
        let key_pair = Ed25519KeyPair::from_pkcs8(pkcs8_bytes.as_ref())
            .map_err(|e| CryptoTensorError::KeyCreation(e.to_string()))?;
        let public_key = key_pair.public_key().as_ref().to_vec();

        // Create test key material for signing
        let sign_key = KeyMaterial::new(
            "okp".to_string(),
            "ed25519".to_string(),
            Some("sig".to_string()),
            Some(vec!["sign".to_string(), "verify".to_string()]),
            Some("test-sign-key".to_string()),
            Some(format!("file://{}", test_file.to_str().unwrap())),
            None,
            Some(public_key.clone()),
            Some(pkcs8_bytes.as_ref().to_vec()),
        )?;

        // Write key material to file
        let jwk_content = serde_json::json!({
            "keys": [
                {
                    "kty": "oct",
                    "use": "enc",
                    "key_ops": ["encrypt", "decrypt"],
                    "alg": "aes256gcm",
                    "kid": "test-enc-key",
                    "k": BASE64.encode(vec![1u8; 32])
                },
                {
                    "kty": "okp",
                    "use": "sig",
                    "key_ops": ["sign", "verify"],
                    "alg": "ed25519",
                    "kid": "test-sign-key",
                    "x": BASE64.encode(public_key),
                    "d": BASE64.encode(pkcs8_bytes.as_ref())
                }
            ]
        });
        fs::write(&test_file, serde_json::to_string_pretty(&jwk_content).unwrap())
            .map_err(|e| CryptoTensorError::KeyLoad { source: e.to_string() })?;

        // Create serialization configuration
        let dummy_policy = LoadPolicy::new(None, None);
        let config = SerializeCryptoConfig::new(
            Some(vec!["tensor1".to_string()]),
            enc_key,
            sign_key,
            dummy_policy,
        )?;

        // Initialize CryptoTensor from serialization config
        let mut crypto_tensor = CryptoTensor::from_serialize_config(
            vec!["tensor1".to_string()],
            &config,
        )?.unwrap();

        // Test data for tensors
        let tensor1_data = b"Test data for tensor1";

        // Encrypt tensor data
        crypto_tensor.silent_encrypt("tensor1", tensor1_data)?;

        // Get encrypted data
        let encrypted_tensor1 = crypto_tensor.get_encrypted_data("tensor1").unwrap();

        // Create tensor info
        let tensors = vec![
            ("tensor1".to_string(), TensorInfo {
                dtype: Dtype::U8,
                shape: vec![1],
                data_offsets: (0, encrypted_tensor1.len()),
            }),
        ];

        // Generate metadata with signature
        let metadata = crypto_tensor.generate_metadata(
            tensors.clone(),
            Some(HashMap::from([
                ("test_key".to_string(), "test_value".to_string()),
            ])),
        )?.unwrap();

        // Create new Metadata with the generated metadata
        let header = Metadata::new(Some(metadata), tensors)?;

        // Create new CryptoTensor from header
        let new_crypto_tensor = CryptoTensor::from_header(&header)?.unwrap();

        // Verify encrypted data can be decrypted
        let decrypted_tensor1 = new_crypto_tensor.silent_decrypt("tensor1", encrypted_tensor1)?;
        assert_eq!(decrypted_tensor1, tensor1_data);
        println!("Successfully decrypted tensor1");

        // Clean up test files
        fs::remove_file(&test_file).unwrap();
        println!("Test completed successfully");

        Ok(())
    }
}