use crate::tensor::{Metadata, TensorInfo};
use once_cell::sync::OnceCell;
use ring::signature::{self, Ed25519KeyPair, UnparsedPublicKey, KeyPair};
use ring::{aead, rand::{self, SecureRandom}};
use serde::{Deserialize, Serialize, de::Error, Deserializer, Serializer};
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
    /// Unsupported version
    VersionUnsupported(String),
    /// Missing version field
    VersionMissing,
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
            CryptoTensorError::VersionUnsupported(version) => {
                write!(f, "Version {} is unsupported", version)
            }
            CryptoTensorError::VersionMissing => {
                write!(f, "Version is missing")
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

/// Serialize and deserialize OnceCell<Option<String>>
mod key_material_serde {
    use super::*;

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

/// Serialize and deserialize OnceCell<String>
mod cryptor_serde {
    use super::*;

    pub fn serialize<S>(cell: &OnceCell<String>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match cell.get() {
            Some(value) => value.serialize(serializer),
            None => serializer.serialize_none(),
        }
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<OnceCell<String>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value: Option<String> = Option::deserialize(deserializer)?;
        let cell = OnceCell::new();
        if let Some(v) = value {
            cell.set(v).map_err(|_| D::Error::custom("Failed to set OnceCell value"))?;
        }
        Ok(cell)
    }
}

/// Key Material structure for managing cryptographic keys
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyMaterial {
    #[serde(rename = "kty")]
    key_type: JwkKeyType,
    
    alg: String,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    kid: Option<String>,
    
    /// The master key encoded in base64 for encryption
    #[serde(skip_serializing, default)]
    #[serde(with = "key_material_serde")]
    pub k: OnceCell<Option<String>>,

    /// The public key encoded in base64 for signing
    #[serde(skip_serializing, default)]
    #[serde(with = "key_material_serde")]
    #[serde(rename = "x")]
    pub x_pub: OnceCell<Option<String>>,

    /// The private key encoded in base64 for signing
    #[serde(skip_serializing, default)]
    #[serde(with = "key_material_serde")]
    #[serde(rename = "d")]
    pub d_priv: OnceCell<Option<String>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    jku: Option<String>,
}

impl KeyMaterial {
    /// Create a new KeyMaterial
    fn new_internal(
        key_type: JwkKeyType,
        alg: String,
        kid: Option<String>,
        jku: Option<String>,
        k: Option<Vec<u8>>,
        x_pub: Option<Vec<u8>>,
        d_priv: Option<Vec<u8>>,
    ) -> Result<Self, CryptoTensorError> {
        let key_material = Self {
            key_type,
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

    /// Create a new symmetric encryption key (kty=oct)
    ///
    /// # Arguments
    /// * `alg` - Optional algorithm name (default: "aes256gcm")
    /// * `kid` - Optional key ID
    /// * `jku` - Optional JWK URL
    /// * `key_b64` - Optional base64-encoded key string
    ///
    /// # Returns
    /// * `Ok(KeyMaterial)` - The generated key material
    /// * `Err(CryptoTensorError)` - If input is invalid
    pub fn new_enc_key(
        key_b64: Option<String>,
        alg: Option<String>,
        kid: Option<String>,
        jku: Option<String>,
    ) -> Result<Self, CryptoTensorError> {
        let alg = alg.unwrap_or_else(|| "aes256gcm".to_string());
        let enc_alg = EncryptionAlgorithm::from_str(&alg)
            .ok_or_else(|| CryptoTensorError::InvalidAlgorithm(alg.clone()))?;
        let key_bytes = if let Some(ref b64_str) = key_b64 {
            let bytes = BASE64.decode(b64_str)
                .map_err(|e| CryptoTensorError::KeyCreation(format!("Failed to decode base64 key: {}", e)))?;
            if bytes.len() != enc_alg.key_len() {
                return Err(CryptoTensorError::InvalidKeyLength {
                    expected: enc_alg.key_len(),
                    got: bytes.len(),
                });
            }
            bytes
        } else {
            // Generate random key
            let mut key = vec![0u8; enc_alg.key_len()];
            let rng = rand::SystemRandom::new();
            rng.fill(&mut key)
                .map_err(|e| CryptoTensorError::RandomGeneration(e.to_string()))?;
            key
        };
        KeyMaterial::new_internal(
            JwkKeyType::Oct,
            alg,
            kid,
            jku,
            Some(key_bytes),
            None,
            None,
        )
    }

    /// Create a new signing key (kty=okp)
    ///
    /// # Arguments
    /// * `alg` - Optional algorithm name (default: "ed25519")
    /// * `kid` - Optional key ID
    /// * `jku` - Optional JWK URL
    /// * `public_b64` - Optional base64-encoded public key string
    /// * `private_b64` - Optional base64-encoded private key string
    ///
    /// # Returns
    /// * `Ok(KeyMaterial)` - The generated key material
    /// * `Err(CryptoTensorError)` - If input is invalid
    pub fn new_sign_key(
        public_b64: Option<String>,
        private_b64: Option<String>,
        alg: Option<String>,
        kid: Option<String>,
        jku: Option<String>,
    ) -> Result<Self, CryptoTensorError> {
        let alg = alg.unwrap_or_else(|| "ed25519".to_string());
        let sig_alg = SignatureAlgorithm::from_str(&alg)
            .ok_or_else(|| CryptoTensorError::InvalidAlgorithm(alg.clone()))?;
        match sig_alg {
            SignatureAlgorithm::Ed25519 => {
                let public = if let Some(pub_b64) = public_b64 {
                    let pub_bytes = BASE64.decode(&pub_b64)
                        .map_err(|e| CryptoTensorError::KeyCreation(format!("Failed to decode base64 public key: {}", e)))?;
                    if pub_bytes.len() != 32 {
                        return Err(CryptoTensorError::InvalidKeyLength {
                            expected: 32,
                            got: pub_bytes.len(),
                        });
                    }
                    Some(pub_bytes)
                } else { None };
                let private = if let Some(priv_b64) = private_b64 {
                    let priv_bytes = BASE64.decode(&priv_b64)
                        .map_err(|e| CryptoTensorError::KeyCreation(format!("Failed to decode base64 private key: {}", e)))?;
                    if priv_bytes.len() != 32 {
                        return Err(CryptoTensorError::InvalidKeyLength {
                            expected: 32,
                            got: priv_bytes.len(),
                        });
                    }
                    Some(priv_bytes)
                } else { None };
                // If both are None, generate new key pair
                let (public, private) = if public.is_none() && private.is_none() {
                    let rng = rand::SystemRandom::new();
                    let mut private_key = [0u8; 32];
                    rng.fill(&mut private_key).map_err(|e| CryptoTensorError::RandomGeneration(e.to_string()))?;
                    let key_pair = Ed25519KeyPair::from_seed_unchecked(&private_key)
                        .map_err(|e| CryptoTensorError::KeyCreation(e.to_string()))?;
                    (Some(key_pair.public_key().as_ref().to_vec()), Some(private_key.to_vec()))
                } else if !public.is_none() && !private.is_none() {
                    // Try to verify the key pair
                    Ed25519KeyPair::from_seed_and_public_key(private.clone().unwrap().as_slice(), public.clone().unwrap().as_slice() )
                        .map_err(|e| CryptoTensorError::KeyCreation(e.to_string()))?;
                    (public, private)
                } else {
                    (public, private)
                };
                KeyMaterial::new_internal(
                    JwkKeyType::Okp,
                    alg,
                    kid,
                    jku,
                    None,
                    public,
                    private,
                )
            }
        }
    }

    /// Convert this KeyMaterial to a JWK JSON string
    ///
    /// # Returns
    /// * `Ok(String)` - JWK JSON string
    /// * `Err(CryptoTensorError)` - If serialization fails
    pub fn to_jwk(&self) -> Result<String, CryptoTensorError> {
        // Only include JWK-relevant fields
        #[derive(Serialize)]
        struct JwkOut<'a> {
            #[serde(rename = "kty")]
            key_type: &'a JwkKeyType,
            alg: &'a String,
            #[serde(skip_serializing_if = "Option::is_none")]
            kid: &'a Option<String>,
            #[serde(skip_serializing_if = "Option::is_none")]
            jku: &'a Option<String>,
            #[serde(skip_serializing_if = "Option::is_none")]
            k: Option<&'a String>,
            #[serde(skip_serializing_if = "Option::is_none")]
            #[serde(rename = "x")]
            x_pub: Option<&'a String>,
            #[serde(skip_serializing_if = "Option::is_none")]
            #[serde(rename = "d")]
            d_priv: Option<&'a String>,
        }
        let jwk = JwkOut {
            key_type: &self.key_type,
            alg: &self.alg,
            kid: &self.kid,
            jku: &self.jku,
            k: self.k.get().and_then(|v| v.as_ref()),
            x_pub: self.x_pub.get().and_then(|v| v.as_ref()),
            d_priv: self.d_priv.get().and_then(|v| v.as_ref()),
        };
        serde_json::to_string(&jwk).map_err(|e| CryptoTensorError::KeyCreation(format!("Failed to serialize JWK: {}", e)))
    }

    /// Parse KeyMaterial from a serde_json::Value (header)
    ///
    /// # Arguments
    /// * `header` - serde_json::Value containing key material
    ///
    /// # Returns
    /// * `Ok<KeyMaterial>` - Key material
    /// * `Err(CryptoTensorError)` - If parsing or validation fails
    fn from_header(header: &serde_json::Value) -> Result<Self, CryptoTensorError> {
        let key: KeyMaterial = serde_json::from_value(header.clone())
            .map_err(|e| CryptoTensorError::InvalidKey(format!("Failed to parse key material: {}", e)))?;
        key.validate(ValidateMode::Load)?;
        Ok(key)
    }
}

/// Configuration for serializing tensors with encryption
pub struct SerializeCryptoConfig {
    /// CryptoTensors version
    version: String,
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
        version: String,
        tensors: Option<Vec<String>>,
        enc_key: KeyMaterial,
        sign_key: KeyMaterial,
        policy: LoadPolicy,
    ) -> Result<Self, CryptoTensorError> {
        if version != "1" {
            return Err(CryptoTensorError::VersionUnsupported(version));
        }
        enc_key.validate(ValidateMode::Save)?;
        sign_key.validate(ValidateMode::Save)?;
        Ok(Self {
            tensors,
            enc_key,
            sign_key,
            policy,
            version,
        })
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
            let key_pair = Ed25519KeyPair::from_seed_unchecked(key).map_err(|e| {
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
pub struct SingleCryptor<'data> {
    /// Algorithm used for encryption
    #[serde(skip)]
    enc_algo: String,
    /// Encrypted tensor key encoded in base64
    #[serde(with = "cryptor_serde")]
    wrapped_key: OnceCell<String>,
    /// Initialization vector for key encryption encoded in base64
    #[serde(with = "cryptor_serde")]
    key_iv: OnceCell<String>,
    /// Authentication tag for key encryption encoded in base64
    #[serde(with = "cryptor_serde")]
    key_tag: OnceCell<String>,
    /// Initialization vector for data encryption encoded in base64
    #[serde(with = "cryptor_serde")]
    iv: OnceCell<String>,
    /// Authentication tag for data encryption encoded in base64
    #[serde(with = "cryptor_serde")]
    tag: OnceCell<String>,
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

impl<'data> SingleCryptor<'data> {
    /// Create a new SingleCryptor from key material
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
                    wrapped_key: OnceCell::new(),
                    key_iv: OnceCell::new(),
                    key_tag: OnceCell::new(),
                    iv: OnceCell::new(),
                    tag: OnceCell::new(),
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
    fn wrap_key(&self, key: &[u8]) -> Result<(), CryptoTensorError> {
        let mut key_buf = key.to_vec();
        let (key_iv, key_tag) =
            encrypt_data(&mut key_buf, &self.master_key, &self.enc_algo)?;
        self.wrapped_key.set(BASE64.encode(&key_buf))
            .map_err(|_| CryptoTensorError::Encryption("Failed to set wrapped key".to_string()))?;
        self.key_iv.set(BASE64.encode(&key_iv))
            .map_err(|_| CryptoTensorError::Encryption("Failed to set key iv".to_string()))?;
        self.key_tag.set(BASE64.encode(&key_tag))
            .map_err(|_| CryptoTensorError::Encryption("Failed to set key tag".to_string()))?;
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

        let mut data_key = BASE64.decode(&self.wrapped_key.get().ok_or_else(|| CryptoTensorError::KeyUnwrap { tensor: "".to_string(), source: "wrapped_key is empty".to_string() })?)
            .map_err(|e| CryptoTensorError::KeyUnwrap { tensor: "".to_string(), source: e.to_string() })?;
        decrypt_data(
            &mut data_key,
            &self.master_key,
            &self.enc_algo,
            BASE64.decode(&self.key_iv.get().ok_or_else(|| CryptoTensorError::KeyUnwrap { tensor: "".to_string(), source: "key_iv is empty".to_string() })?)
                .map_err(|e| CryptoTensorError::KeyUnwrap { tensor: "".to_string(), source: e.to_string() })?.as_slice(),
            BASE64.decode(&self.key_tag.get().ok_or_else(|| CryptoTensorError::KeyUnwrap { tensor: "".to_string(), source: "key_tag is empty".to_string() })?)
                .map_err(|e| CryptoTensorError::KeyUnwrap { tensor: "".to_string(), source: e.to_string() })?.as_slice(),
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
    fn decrypt(&'data self, data: &[u8]) -> Result<&'data [u8], CryptoTensorError> {
        self.buffer
            .get_or_try_init(|| {
                let data_key = Zeroizing::new(self.unwrap_key()?);

                let mut buffer = data.to_vec();
                decrypt_data(
                    &mut buffer,
                    data_key.as_slice(),
                    &self.enc_algo,
                    BASE64.decode(&self.iv.get().ok_or_else(|| CryptoTensorError::KeyUnwrap { tensor: "".to_string(), source: "iv is empty".to_string() })?)
                        .map_err(|e| CryptoTensorError::KeyUnwrap { tensor: "".to_string(), source: e.to_string() })?.as_slice(),
                    BASE64.decode(&self.tag.get().ok_or_else(|| CryptoTensorError::KeyUnwrap { tensor: "".to_string(), source: "tag is empty".to_string() })?)
                        .map_err(|e| CryptoTensorError::KeyUnwrap { tensor: "".to_string(), source: e.to_string() })?.as_slice(),
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
    fn encrypt(&self, data: &[u8]) -> Result<(), CryptoTensorError> {
        // Generate random data encryption key
        let data_key = Zeroizing::new(self.random_key()?);
        // Copy data to buffer, prepare in-place encryption
        let mut buffer = data.to_vec();
        let (iv, tag) = encrypt_data(&mut buffer, &data_key, &self.enc_algo)?;
        self.iv.set(BASE64.encode(&iv))
            .map_err(|_| CryptoTensorError::Encryption("Failed to set iv".to_string()))?;
        self.tag.set(BASE64.encode(&tag))
            .map_err(|_| CryptoTensorError::Encryption("Failed to set tag".to_string()))?;
        self.wrap_key(&data_key)?;
        self.buffer.set(buffer)
            .map_err(|_| CryptoTensorError::Encryption("Failed to set buffer".to_string()))?;
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
pub struct CryptoTensors<'data> {
    /// Mapping from tensor names to their encryptors
    cryptors: HashMap<String, SingleCryptor<'data>>,
    /// Signer for signing/verifying the file header
    signer: HeaderSigner,
    /// Key material for encryption/decryption
    enc_key: KeyMaterial,
    /// Key material for signing/verification
    sign_key: KeyMaterial,
    /// Policy for model loading and KMS validation
    policy: LoadPolicy,
    /// CryptoTensors version
    version: String,
}

impl<'data> CryptoTensors<'data> {
    /// Get the encryptor for a specific tensor
    pub fn get(&self, tensor_name: &str) -> Option<&SingleCryptor<'data>> {
        self.cryptors.get(tensor_name)
    }

    /// Return bool if the tensor should be encrypted
    pub fn should_encrypt(&self, tensor_name: String) -> bool {
        self.cryptors.contains_key(&tensor_name)
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
    /// A new CryptoTensors instance. If no configuration is provided or no tensors
    /// are selected for encryption, the manager will be initialized without any
    /// encryptors.
    pub fn from_serialize_config(
        tensors: Vec<String>,
        config: &SerializeCryptoConfig,
    ) -> Result<Option<Self>, CryptoTensorError> {
        // Check version
        if config.version != "1" {
            return Err(CryptoTensorError::VersionUnsupported(config.version.clone()));
        }
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
                let cryptor = SingleCryptor::new(&config.enc_key)?;
                Ok((name.clone(), cryptor))
            })
            .collect::<Result<HashMap<String, SingleCryptor<'data>>, CryptoTensorError>>()?;

        // Create signer
        let signer = HeaderSigner::new(&config.sign_key)?;

        Ok(Some(Self {
            cryptors,
            enc_key: config.enc_key.clone(),
            sign_key: config.sign_key.clone(),
            signer,
            policy: config.policy.clone(),
            version: config.version.clone(),
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
            "version": self.version,
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
        new_metadata.insert("__signature__".to_string(), BASE64.encode(signature));

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
    /// * `Ok(CryptoTensors)` - If valid encryption metadata was found and verified
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
        let version = key_materials.get("version")
            .and_then(|v| v.as_str())
            .ok_or(CryptoTensorError::VersionMissing)?;
        if version != "1" {
            return Err(CryptoTensorError::VersionUnsupported(version.to_string()));
        }
        let enc_key = KeyMaterial::from_header(&key_materials["enc"])?;
        let sign_key = KeyMaterial::from_header(&key_materials["sign"])?;


        // Load keys
        sign_key.load_key()?;

        // Create signer and verify signature
        let signer = HeaderSigner::new(&sign_key)?;
        let signature = BASE64.decode(signature_hex)
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

        let mut cryptors: HashMap<String, SingleCryptor<'data>> = serde_json::from_str(encryption_info)
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
            version: version.to_string(),
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
        &self,
        tensor_name: &str,
        data: &[u8],
    ) -> Result<(), CryptoTensorError> {
        match self.get(tensor_name) {
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
    pub fn get_buffer(&self, tensor_name: &str) -> Option<&[u8]> {
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
        let key_material = KeyMaterial::new_enc_key(
            Some(BASE64.encode(master_key)),
            Some("aes256gcm".to_string()),
            None,
            None,
        )?;

        let encryptor = SingleCryptor::new(&key_material)?;
        let empty_data = b"";
        assert!(encryptor.encrypt(empty_data).is_ok());
        let encrypted_empty = encryptor.buffer.get().unwrap();
        assert!(encrypted_empty.is_empty()); // Should be empty encrypted data

        let mut decryptor = SingleCryptor::new(&key_material)?;
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
        let key_material = KeyMaterial::new_enc_key(
            Some(BASE64.encode(master_key)),
            Some("aes256gcm".to_string()),
            None,
            None,
        )?;

        let encryptor = SingleCryptor::new(&key_material)?;
        let test_data = b"Hello";
        assert!(encryptor.encrypt(test_data).is_ok());
        let encrypted_data = encryptor.buffer.get().unwrap();
        assert_ne!(encrypted_data, test_data);

        let mut decryptor = SingleCryptor::new(&key_material)?;
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
        let key_material = KeyMaterial::new_enc_key(
            Some(BASE64.encode(master_key)),
            Some("aes256gcm".to_string()),
            None,
            None,
        )?;

        let encryptor = SingleCryptor::new(&key_material)?;
        let large_data = vec![1u8; 1024];
        assert!(encryptor.encrypt(&large_data).is_ok());
        let encrypted_data = encryptor.buffer.get().unwrap();
        assert_ne!(encrypted_data, &large_data);

        let mut decryptor = SingleCryptor::new(&key_material)?;
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
            let key_material = KeyMaterial::new_enc_key(
                Some(BASE64.encode(master_key)),
                Some(key_algo.to_string()),
                None,
                None,
            )?;

            let encryptor = SingleCryptor::new(&key_material)?;
            assert!(encryptor.encrypt(test_data).is_ok());
            let encrypted_data = encryptor.buffer.get().unwrap();
            assert_ne!(encrypted_data, test_data);

            let mut decryptor = SingleCryptor::new(&key_material)?;
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
        let mut private_key = [0u8; 32];
        rng.fill(&mut private_key).map_err(|e| CryptoTensorError::RandomGeneration(e.to_string()))?;
        let key_pair = Ed25519KeyPair::from_seed_unchecked(&private_key)
            .map_err(|e| CryptoTensorError::KeyCreation(e.to_string()))?;
        let public_key = key_pair.public_key().as_ref().to_vec();

        // Create key material
        let key_material = KeyMaterial::new_sign_key(
            Some(BASE64.encode(public_key)),
            Some(BASE64.encode(private_key)),
            Some("ed25519".to_string()),
            None,
            None,
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
        // Generate Ed25519 key pair
        let rng = SystemRandom::new();
        let mut private_key = [0u8; 32];
        rng.fill(&mut private_key).map_err(|e| CryptoTensorError::RandomGeneration(e.to_string()))?;
        let key_pair = Ed25519KeyPair::from_seed_unchecked(&private_key)
            .map_err(|e| CryptoTensorError::KeyCreation(e.to_string()))?;
        let public_key = key_pair.public_key().as_ref().to_vec();

        // Test missing private key
        let key_material = KeyMaterial::new_sign_key(
            Some(BASE64.encode(public_key)),
            None, // No private key
            Some("ed25519".to_string()),
            None,
            None,
        )?;

        let signer = HeaderSigner::new(&key_material)?;
        let test_data = b"Test data";
        
        // Attempt to sign should fail
        assert!(matches!(
            signer.sign(test_data),
            Err(CryptoTensorError::MissingSigningKey)
        ));

        // Test missing public key
        let key_material = KeyMaterial::new_sign_key(
            None,
            Some(BASE64.encode(&private_key)),
            Some("ed25519".to_string()),
            None,
            None,
        )?;

        let signer = HeaderSigner::new(&key_material)?;
        
        // Attempt to verify should fail
        assert!(matches!(
            signer.verify(test_data),
            Err(CryptoTensorError::MissingVerificationKey)
        ));

        // Test invalid signature algorithm
        assert!(matches!(
            KeyMaterial::new_sign_key(
                None,
                None,
                Some("invalid_algorithm".to_string()),
                None,
                None,
            ),
            Err(CryptoTensorError::InvalidAlgorithm(_))
        ));

        Ok(())
    }

    /// Test if two repetitive signing will generate the same signature
    #[test]
    fn test_header_signer_signature_repeat() -> Result<(), CryptoTensorError> {
        // Create key material (random key pair)
        let key_material = KeyMaterial::new_sign_key(
            None,
            None,
            None,
            None,
            None,
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
        // Test default encryption key creation
        let enc_key = KeyMaterial::new_enc_key(None, None, None, None).unwrap();
        assert_eq!(enc_key.key_type, JwkKeyType::Oct);
        assert_eq!(enc_key.alg, "aes256gcm");

        // Test default signing key creation
        let sign_key = KeyMaterial::new_sign_key(None, None, None, None, None).unwrap();
        assert_eq!(sign_key.key_type, JwkKeyType::Okp);
        assert_eq!(sign_key.alg, "ed25519");

        // Test validation in Save mode
        // Test missing jku in Save mode (should pass)
        let save_key = KeyMaterial::new_enc_key(
            None,
            Some("aes256gcm".to_string()),
            Some("test-key".to_string()),
            None, // Missing jku
        ).unwrap();
        assert!(save_key.validate(ValidateMode::Save).is_ok());

        // Test missing kid, jku and key in Load mode (should pass)
        let header = serde_json::json!({
            "kty": "oct",
            "alg": "aes256gcm",
        });
        assert!(KeyMaterial::from_header(&header).is_ok());

        // Test missing algorithm in Load mode (should error)
        let header = serde_json::json!({
            "kty": "oct",
        });
        assert!(matches!(
            KeyMaterial::from_header(&header),
            Err(CryptoTensorError::InvalidKey(_))
        ));

        // Test invalid algorithm in Load mode (should error)
        let header = serde_json::json!({
            "kty": "oct",
            "alg": "invalid_algorithm",
        });
        assert!(matches!(
            KeyMaterial::from_header(&header),
            Err(CryptoTensorError::InvalidAlgorithm(_))
        ));
    }

    /// Test KeyMaterial key loading functionality
    #[test]
    fn test_key_material_load_single_key_from_jku() -> Result<(), CryptoTensorError> {
        // Create temporary directory and test file
        let temp_dir = tempfile::tempdir().unwrap();
        let test_file = temp_dir.path().join("test.jwk");

        // Write single key JWK file
        let key = KeyMaterial::new_enc_key(None, Some("aes256gcm".to_string()), Some("test-key".to_string()), None).unwrap();
        let jwk_content = key.to_jwk().unwrap();
        fs::write(&test_file, jwk_content)
            .map_err(|e| CryptoTensorError::KeyLoad { source: e.to_string() })?;

        // Create KeyMaterial and attempt to load key
        let header = serde_json::json!({
            "kty": "oct",
            "alg": "aes256gcm",
            "kid": "test-key",
            "jku": format!("file://{}", test_file.to_str().unwrap()),
        });
        let key_material = KeyMaterial::from_header(&header)?;

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

        // Create JWK Set file content
        let key1 = KeyMaterial::new_enc_key(None, Some("aes256gcm".to_string()), Some("key1".to_string()), None).unwrap();
        let key2 = KeyMaterial::new_enc_key(None, Some("aes256gcm".to_string()), Some("key2".to_string()), None).unwrap();
        let jwk_set_content = format!(
            r#"{{
                "keys": [
                    {},
                    {}
                ]
            }}"#, key1.to_jwk().unwrap(), key2.to_jwk().unwrap());

        // Write to file
        fs::write(&test_file, jwk_set_content)
            .map_err(|e| CryptoTensorError::KeyLoad { source: e.to_string() })?;

        // Test loading first key
        let header = serde_json::json!({
            "kty": "oct",
            "alg": "aes256gcm",
            "kid": "key1",
            "jku": format!("file://{}", test_file.to_str().unwrap()),
        });
        let key_material = KeyMaterial::from_header(&header)?;

        // Load key
        let load_result = key_material.load_key();
        assert!(load_result.is_ok(), "Failed to load encryption key: {}", load_result.unwrap_err());
        assert!(key_material.k.get().is_some());

        // Test loading second key
        let header = serde_json::json!({
            "kty": "oct",
            "alg": "aes256gcm",
            "kid": "key2",
            "jku": format!("file://{}", test_file.to_str().unwrap()),
        });
        let key_material = KeyMaterial::from_header(&header)?;

        // Load key
        let load_result = key_material.load_key();
        assert!(load_result.is_ok(), "Failed to load encryption key: {}", load_result.unwrap_err());
        assert!(key_material.k.get().is_some());

        // Test loading non-existent key
        let header = serde_json::json!({
            "kty": "oct",
            "alg": "aes256gcm",
            "kid": "non-existent",
            "jku": format!("file://{}", test_file.to_str().unwrap()),
        });
        let key_material = KeyMaterial::from_header(&header)?;

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
        let key1 = KeyMaterial::new_enc_key(None, Some("aes256gcm".to_string()), Some("key1".to_string()), None).unwrap();
        let key2 = KeyMaterial::new_sign_key(None, None, Some("ed25519".to_string()), Some("key2".to_string()), None).unwrap();
        let jwk_set_content = format!(
            r#"{{
                "keys": [
                    {},
                    {}
                ]
            }}"#, key1.to_jwk().unwrap(), key2.to_jwk().unwrap());

        // Write to file
        fs::write(&test_file, jwk_set_content)
            .map_err(|e| CryptoTensorError::KeyLoad { source: e.to_string() })?;

        // Set environment variable
        let file_path = format!("file://{}", test_file.to_str().unwrap());
        std::env::set_var("CRYPTOTENSOR_KEY_JKU", file_path);

        // Test loading encryption key
        let header = serde_json::json!({
            "kty": "oct",
            "alg": "aes256gcm",
            "kid": "key1",
        });
        let enc_key = KeyMaterial::from_header(&header)?;

        // Load key
        let load_result = enc_key.load_key();
        assert!(load_result.is_ok(), "Failed to load encryption key: {}", load_result.unwrap_err());
        assert!(enc_key.k.get().is_some());
        assert!(enc_key.validate(ValidateMode::Load).is_ok());

        // Test loading signing key
        let header = serde_json::json!({
            "kty": "okp",
            "alg": "ed25519",
            "kid": "key2",
        });
        let sign_key = KeyMaterial::from_header(&header)?;

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
        let key1 = KeyMaterial::new_enc_key(None, Some("aes256gcm".to_string()), Some("key1".to_string()), None).unwrap();
        let key2 = KeyMaterial::new_sign_key(None, None, Some("ed25519".to_string()), Some("key2".to_string()), None).unwrap();
        let jwk_set_content = format!(
            r#"{{
                "keys": [
                    {},
                    {}
                ]
            }}"#, key1.to_jwk().unwrap(), key2.to_jwk().unwrap());
        fs::write(&default_file, jwk_set_content)
            .map_err(|e| CryptoTensorError::KeyLoad { source: e.to_string() })?;

        // Test loading encryption key
        let header = serde_json::json!({
            "kty": "oct",
            "alg": "aes256gcm",
            "kid": "key1",
        });
        let enc_key = KeyMaterial::from_header(&header)?;

        // Load key
        let load_result = enc_key.load_key();
        assert!(load_result.is_ok(), "Failed to load encryption key: {}", load_result.unwrap_err());
        assert!(enc_key.k.get().is_some());
        assert!(enc_key.validate(ValidateMode::Load).is_ok());

        // Test loading signing key
        let header = serde_json::json!({
            "kty": "okp",
            "alg": "ed25519",
            "kid": "key2",
        });
        let sign_key = KeyMaterial::from_header(&header)?;

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

    /// Format JSON value, supports nested structure
    fn format_json_value(value: &serde_json::Value, indent: usize, key: Option<&str>) -> String {
        const BYTE_FIELDS: &[&str] = &["iv", "tag", "key_iv", "key_tag", "wrapped_key"];
        match value {
            serde_json::Value::Object(map) => {
                let indent_str = " ".repeat(indent);
                let mut result = String::from("{\n");
                let mut first = true;
                for (k, val) in map {
                    if !first {
                        result.push_str(",\n");
                    }
                    first = false;
                    result.push_str(&format!(
                        "{}  \"{}\": {}",
                        indent_str,
                        k,
                        format_json_value(val, indent + 2, Some(k))
                    ));
                }
                format!("{}\n{}}}", result, indent_str)
            }
            serde_json::Value::Array(arr) => {
                if let Some(k) = key {
                    if BYTE_FIELDS.contains(&k) {
                        return format!("[{} bytes array]", arr.len());
                    }
                }
                if arr.len() > 16 && arr.iter().all(|v| v.is_number()) {
                    format!("[{} bytes array]", arr.len())
                } else {
                    let indent_str = " ".repeat(indent);
                    let mut result = String::from("[\n");
                    let mut first = true;
                    for val in arr {
                        if !first {
                            result.push_str(",\n");
                        }
                        first = false;
                        result.push_str(&format!(
                            "{}  {}",
                            indent_str,
                            format_json_value(val, indent + 2, None)
                        ));
                    }
                    format!("{}\n{}]", result, indent_str)
                }
            }
            serde_json::Value::String(s) => {
                if let Ok(json_val) = serde_json::from_str::<serde_json::Value>(s) {
                    format_json_value(&json_val, indent, None)
                } else if s.contains('\n') {
                    format!("\"{}\"", s.replace('\n', "\\n"))
                } else {
                    format!("\"{}\"", s)
                }
            }
            _ => value.to_string(),
        }
    }

    /// Print formatted JSON object
    fn print_formatted_json(title: &str, value: &serde_json::Value) {
        println!("\n=== {} ===", title);
        println!("{}", format_json_value(value, 0, None));
        println!("=== End of {} ===\n", title);
    }

    /// Test the complete flow of CryptoTensors
    #[test]
    fn test_crypto_tensor_complete_flow() -> Result<(), CryptoTensorError> {
        // Create temporary directory for test files
        let temp_dir = tempfile::tempdir().unwrap();
        let test_file = temp_dir.path().join("test.jwk");

        // Create test key material for encryption
        let enc_key = KeyMaterial::new_enc_key(
            Some(BASE64.encode(vec![1u8; 32])),
            Some("aes256gcm".to_string()),
            Some("test-enc-key".to_string()),
            Some(format!("file://{}", test_file.to_str().unwrap())),
        )?;

        // Generate Ed25519 key pair for signing
        let rng = SystemRandom::new();
        let mut private_key = [0u8; 32];
        rng.fill(&mut private_key).map_err(|e| CryptoTensorError::RandomGeneration(e.to_string()))?;
        let key_pair = Ed25519KeyPair::from_seed_unchecked(&private_key)
            .map_err(|e| CryptoTensorError::KeyCreation(e.to_string()))?;
        let public_key = key_pair.public_key().as_ref().to_vec();

        // Create test key material for signing
        let sign_key = KeyMaterial::new_sign_key(
            Some(BASE64.encode(public_key.clone())),
            Some(BASE64.encode(&private_key)),
            Some("ed25519".to_string()),
            Some("test-sign-key".to_string()),
            Some(format!("file://{}", test_file.to_str().unwrap())),
        )?;

        // Write key material to file
        let jwk_content = format!(
            r#"{{
                "keys": [
                    {},
                    {}
                ]
            }}"#, enc_key.to_jwk().unwrap(), sign_key.to_jwk().unwrap());
        fs::write(&test_file, jwk_content)
            .map_err(|e| CryptoTensorError::KeyLoad { source: e.to_string() })?;

        // Create serialization configuration
        let dummy_policy = LoadPolicy::new(None, None);
        let config = SerializeCryptoConfig::new(
            "1".to_string(),
            Some(vec!["tensor1".to_string()]),
            enc_key,
            sign_key,
            dummy_policy,
        )?;

        // Initialize CryptoTensors from serialization config
        let crypto_tensor = CryptoTensors::from_serialize_config(
            vec!["tensor1".to_string()],
            &config,
        )?.unwrap();

        // Test data for tensors
        let tensor1_data = b"Test data for tensor1";

        // Encrypt tensor data
        crypto_tensor.silent_encrypt("tensor1", tensor1_data)?;

        // Get encrypted data
        let encrypted_tensor1 = crypto_tensor.get_buffer("tensor1").unwrap();

        // Create tensor info
        let tensors = vec![
            ("tensor1".to_string(), TensorInfo {
                dtype: Dtype::U8,
                shape: vec![encrypted_tensor1.len()],
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

        // Print formatted complete header
        let header_json = serde_json::to_value(&header).unwrap();
        print_formatted_json("Complete Header", &header_json);

        // Create new CryptoTensors from header
        let new_crypto_tensor = CryptoTensors::from_header(&header)?.unwrap();

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