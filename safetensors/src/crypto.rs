use crate::lib::HashMap;
use serde::{Deserialize, Serialize};
use ring::{aead, rand::{self, SecureRandom}, pbkdf2};
use std::num::NonZeroU32;

/// Supported encryption algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncryptionAlgorithm {
    /// AES-128-GCM
    Aes128Gcm,
    /// AES-256-GCM
    Aes256Gcm,
    /// ChaCha20-Poly1305
    ChaCha20Poly1305,
}

impl EncryptionAlgorithm {
    /// Convert from string to encryption algorithm
    pub fn from_str(s: &str) -> Option<Self> {
        // Remove hyphens and convert to lowercase for simplified matching
        let normalized = s.replace('-', "").to_lowercase();
        
        match normalized.as_str() {
            "aes128gcm" => Some(EncryptionAlgorithm::Aes128Gcm),
            "aes256gcm" => Some(EncryptionAlgorithm::Aes256Gcm),
            "chacha20poly1305" => Some(EncryptionAlgorithm::ChaCha20Poly1305),
            _ => None,
        }
    }

    /// Convert to string representation
    pub fn to_string(&self) -> String {
        match self {
            EncryptionAlgorithm::Aes128Gcm => "aes128gcm".to_string(),
            EncryptionAlgorithm::Aes256Gcm => "aes256gcm".to_string(),
            EncryptionAlgorithm::ChaCha20Poly1305 => "chacha20poly1305".to_string(),
        }
    }

    /// Get the appropriate aead algorithm from ring
    pub fn get_aead_algo(&self) -> &'static aead::Algorithm {
        match self {
            EncryptionAlgorithm::Aes128Gcm => &aead::AES_128_GCM,
            EncryptionAlgorithm::Aes256Gcm => &aead::AES_256_GCM,
            EncryptionAlgorithm::ChaCha20Poly1305 => &aead::CHACHA20_POLY1305,
        }
    }
    
    /// Get the required key length in bytes
    pub fn key_len(&self) -> usize {
        match self {
            EncryptionAlgorithm::Aes128Gcm => 16, // 128 bits
            EncryptionAlgorithm::Aes256Gcm => 32, // 256 bits
            EncryptionAlgorithm::ChaCha20Poly1305 => 32, // 256 bits
        }
    }
}

/// Information about encrypted tensor data
#[derive(Debug, PartialEq, Eq, Clone, Serialize, Deserialize)]
pub struct TensorEncryptionInfo {
    /// Algorithm used to encrypt the key
    pub key_enc_algo: String,
    /// Algorithm used to encrypt the data
    pub data_enc_algo: String,
    /// Key identifier
    pub key_id: String,
    /// Encrypted key
    pub enc_key: Vec<u8>,
    /// Initialization vector for key encryption
    pub key_iv: Vec<u8>,
    /// Authentication tag for key encryption
    pub key_tag: Vec<u8>,
    /// Initialization vector for data encryption
    pub iv: Vec<u8>,
    /// Authentication tag for data encryption
    pub tag: Vec<u8>,
}

impl TensorEncryptionInfo {
    /// Create a new TensorEncryptionInfo
    pub fn new(
        key_enc_algo: String,
        data_enc_algo: String,
        key_id: String,
        enc_key: Vec<u8>,
        key_iv: Vec<u8>,
        key_tag: Vec<u8>,
        iv: Vec<u8>,
        tag: Vec<u8>,
    ) -> Self {
        Self {
            key_enc_algo,
            data_enc_algo,
            key_id,
            enc_key,
            key_iv,
            key_tag,
            iv,
            tag,
        }
    }

    /// Create an empty TensorEncryptionInfo
    pub fn empty(key_enc_algo: String, data_enc_algo: String, key_id: String) -> Self {
        Self {
            key_enc_algo,
            data_enc_algo,
            key_id,
            enc_key: vec![],
            key_iv: vec![],
            key_tag: vec![],
            iv: vec![],
            tag: vec![],
        }
    }

    /// Validate the encryption algorithms
    pub fn validate_algorithms(&self) -> Result<(), String> {
        EncryptionAlgorithm::from_str(&self.key_enc_algo)
            .ok_or_else(|| format!("Unsupported key encryption algorithm: {}", self.key_enc_algo))?;
        
        EncryptionAlgorithm::from_str(&self.data_enc_algo)
            .ok_or_else(|| format!("Unsupported data encryption algorithm: {}", self.data_enc_algo))?;
        
        Ok(())
    }
}

/// Encryption configuration structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorEncryptionConfig {
    /// Master key used to encrypt individual tensor keys
    pub master_key: Vec<u8>,
    /// Algorithm used to encrypt the keys
    pub key_enc_algo: String,
    /// Algorithm used to encrypt the tensor data
    pub data_enc_algo: String,
    /// Optional key identifier
    pub key_id: Option<String>,
    /// List of tensor names to encrypt, if empty all tensors will be encrypted
    pub encrypted_tensors: Vec<String>,
}

impl TensorEncryptionConfig {
    /// Create a new encryption configuration
    pub fn new(
        master_key: Vec<u8>,
        key_enc_algo: String,
        data_enc_algo: String,
        key_id: Option<String>,
        encrypted_tensors: Vec<String>,
    ) -> Self {
        Self {
            master_key,
            key_enc_algo,
            data_enc_algo,
            key_id,
            encrypted_tensors,
        }
    }

    /// Check if the specified tensor should be encrypted
    pub fn should_encrypt_tensor(&self, tensor_name: &str) -> bool {
        self.encrypted_tensors.is_empty() || self.encrypted_tensors.contains(&tensor_name.to_string())
    }

    /// Generate encryption information for a specific tensor
    pub fn generate_enc_info(&self, tensor_name: &str) -> Option<TensorEncryptionInfo> {
        if !self.should_encrypt_tensor(tensor_name) {
            return None;
        }

        // Parse encryption algorithms
        let key_algo = EncryptionAlgorithm::from_str(&self.key_enc_algo)?;
        let data_algo = EncryptionAlgorithm::from_str(&self.data_enc_algo)?;
        
        // Generate a random salt for key derivation
        let rng = rand::SystemRandom::new();
        let mut salt = vec![0u8; 16];
        rng.fill(&mut salt).unwrap();
        
        // Derive a key for encrypting the tensor key
        let derived_key = derive_key(&self.master_key, &salt, 10000, key_algo.key_len());
        
        // Generate a random tensor key
        let mut tensor_key = vec![0u8; data_algo.key_len()];
        rng.fill(&mut tensor_key).unwrap();
        
        // Create a temporary TensorEncryptionInfo for key encryption
        let mut temp_info = TensorEncryptionInfo::empty(
            self.key_enc_algo.clone(),
            self.data_enc_algo.clone(),
            self.key_id.clone().unwrap_or_else(|| format!("key_{}", tensor_name)),
        );
        temp_info.enc_key = derived_key;
        
        // Encrypt the tensor key
        let enc_key = encrypt_data(&tensor_key, &mut temp_info).ok()?;
        
        // Create the final TensorEncryptionInfo
        Some(TensorEncryptionInfo::new(
            self.key_enc_algo.clone(),
            self.data_enc_algo.clone(),
            temp_info.key_id,
            enc_key,
            temp_info.iv,
            temp_info.tag,
            vec![],
            vec![],
        ))
    }

    /// Generate encryption information for all tensors that need to be encrypted
    pub fn generate_all_enc_info(&self, tensor_names: &[String]) -> HashMap<String, TensorEncryptionInfo> {
        tensor_names
            .iter()
            .filter_map(|name| self.generate_enc_info(name).map(|info| (name.clone(), info)))
            .collect()
    }
}

/// Derive a key from a master key and salt using PBKDF2
/// 
/// # Arguments
/// 
/// * `master_key` - The master key to derive from
/// * `salt` - Random salt for key derivation
/// * `iterations` - Number of iterations for PBKDF2
/// * `key_len` - Length of the derived key in bytes
/// 
/// # Returns
/// 
/// A derived key of the specified length
pub fn derive_key(master_key: &[u8], salt: &[u8], iterations: u32, key_len: usize) -> Vec<u8> {
    let mut derived_key = vec![0u8; key_len];
    let iterations = NonZeroU32::new(iterations).unwrap_or(NonZeroU32::new(10000).unwrap());
    
    pbkdf2::derive(
        pbkdf2::PBKDF2_HMAC_SHA256,
        iterations,
        salt,
        master_key,
        &mut derived_key,
    );
    
    derived_key
}

/// Encrypt data using the specified algorithm
/// 
/// # Arguments
/// 
/// * `data` - The data to encrypt
/// * `enc_info` - The encryption information
/// 
/// # Returns
/// 
/// The encrypted data
pub fn encrypt_data(data: &[u8], enc_info: &mut TensorEncryptionInfo) -> Result<Vec<u8>, String> {
    // Validate the encryption algorithm
    enc_info.validate_algorithms()?;
    
    let algo = EncryptionAlgorithm::from_str(&enc_info.data_enc_algo)
        .ok_or_else(|| format!("Unsupported encryption algorithm: {}", enc_info.data_enc_algo))?;
    
    let aead_algo = algo.get_aead_algo();
    let key = aead::UnboundKey::new(aead_algo, &enc_info.enc_key)
        .map_err(|e| format!("Failed to create unbound key: {}", e))?;
    let key = aead::LessSafeKey::new(key);
    
    // Generate a new nonce
    let mut nonce_bytes = vec![0u8; aead_algo.nonce_len()];
    let rng = rand::SystemRandom::new();
    rng.fill(&mut nonce_bytes)
        .map_err(|e| format!("Failed to generate nonce: {}", e))?;
    let nonce = aead::Nonce::assume_unique_for_key(nonce_bytes.try_into().unwrap());
    
    // Store the nonce in the encryption info
    enc_info.iv = nonce.as_ref().to_vec();
    
    // Encrypt the data
    let mut in_out = data.to_vec();
    key.seal_in_place_append_tag(nonce, aead::Aad::empty(), &mut in_out)
        .map_err(|e| format!("Failed to encrypt data: {}", e))?;
    
    // Extract the tag from the end of the encrypted data
    let tag_len = aead_algo.tag_len();
    let (data, tag) = in_out.split_at(in_out.len() - tag_len);
    enc_info.tag = tag.to_vec();
    
    Ok(data.to_vec())
}

/// Decrypt data using the specified algorithm
/// 
/// # Arguments
/// 
/// * `encrypted_data` - The encrypted data
/// * `enc_info` - The encryption information
/// 
/// # Returns
/// 
/// The decrypted data or an error if decryption failed
pub fn decrypt_data(encrypted_data: &[u8], enc_info: &TensorEncryptionInfo) -> Result<Vec<u8>, String> {
    // Validate the encryption algorithm
    enc_info.validate_algorithms()?;
    
    let algo = EncryptionAlgorithm::from_str(&enc_info.data_enc_algo)
        .ok_or_else(|| format!("Unsupported encryption algorithm: {}", enc_info.data_enc_algo))?;
    
    let aead_algo = algo.get_aead_algo();
    let key = aead::UnboundKey::new(aead_algo, &enc_info.enc_key)
        .map_err(|e| format!("Failed to create unbound key: {}", e))?;
    let key = aead::LessSafeKey::new(key);
    
    // Create nonce from stored IV
    let nonce = aead::Nonce::try_assume_unique_for_key(&enc_info.iv)
        .map_err(|e| format!("Invalid nonce: {}", e))?;
    
    // Combine encrypted data with tag
    let mut in_out = encrypted_data.to_vec();
    in_out.extend_from_slice(&enc_info.tag);
    
    // Decrypt the data
    let decrypted = key.open_in_place(nonce, aead::Aad::empty(), &mut in_out)
        .map_err(|e| format!("Failed to decrypt data: {}", e))?;
    
    Ok(decrypted.to_vec())
}

/// Parse encryption information from a string
pub fn parse_encryption_info(enc_str: &str) -> Result<HashMap<String, TensorEncryptionInfo>, serde_json::Error> {
    serde_json::from_str(enc_str)
}

/// Serialize encryption information to a string
pub fn serialize_encryption_info(enc_info: &HashMap<String, TensorEncryptionInfo>) -> Result<String, serde_json::Error> {
    serde_json::to_string(enc_info)
}

/// Generate encryption information based on encryption configuration and tensor names
pub fn generate_encryption_info(
    enc_config: &TensorEncryptionConfig,
    tensor_names: &[String],
) -> Option<HashMap<String, TensorEncryptionInfo>> {
    if enc_config.master_key.is_empty() {
        return None;
    }
    
    Some(enc_config.generate_all_enc_info(tensor_names))
}

/// Encrypt tensor data using the provided encryption information
/// 
/// # Arguments
/// 
/// * `data` - The tensor data to encrypt
/// * `enc_info` - The encryption information
/// 
/// # Returns
/// 
/// The encrypted data or an error if encryption failed
pub fn encrypt_tensor_data(data: &[u8], enc_info: &mut TensorEncryptionInfo) -> Result<Vec<u8>, String> {
    // Validate the encryption algorithm
    enc_info.validate_algorithms()?;
    
    // Create a temporary TensorEncryptionInfo for data encryption
    let mut data_enc_info = TensorEncryptionInfo::new(
        enc_info.key_enc_algo.clone(),
        enc_info.data_enc_algo.clone(),
        enc_info.key_id.clone(),
        enc_info.enc_key.clone(),
        enc_info.key_iv.clone(),
        enc_info.key_tag.clone(),
        vec![],
        vec![],
    );
    
    // Encrypt the data
    let encrypted_data = encrypt_data(data, &mut data_enc_info)?;
    
    // Update the original enc_info with the data encryption IV and tag
    enc_info.iv = data_enc_info.iv;
    enc_info.tag = data_enc_info.tag;
    
    Ok(encrypted_data)
}

/// Decrypt tensor data using the provided encryption information
/// 
/// # Arguments
/// 
/// * `encrypted_data` - The encrypted tensor data
/// * `enc_info` - The encryption information
/// 
/// # Returns
/// 
/// The decrypted data or an error if decryption failed
pub fn decrypt_tensor_data(encrypted_data: &[u8], enc_info: &TensorEncryptionInfo) -> Result<Vec<u8>, String> {
    // Validate the encryption algorithm
    enc_info.validate_algorithms()?;
    
    // Decrypt the data using the provided encryption info
    decrypt_data(encrypted_data, enc_info)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_encryption_algorithms() {
        // Test conversion from string to algorithm
        assert_eq!(
            EncryptionAlgorithm::from_str("aes-128-gcm"),
            Some(EncryptionAlgorithm::Aes128Gcm)
        );
        assert_eq!(
            EncryptionAlgorithm::from_str("aes128gcm"),
            Some(EncryptionAlgorithm::Aes128Gcm)
        );
        assert_eq!(
            EncryptionAlgorithm::from_str("AES128GCM"),
            Some(EncryptionAlgorithm::Aes128Gcm)
        );
        assert_eq!(
            EncryptionAlgorithm::from_str("aes-256-gcm"),
            Some(EncryptionAlgorithm::Aes256Gcm)
        );
        assert_eq!(
            EncryptionAlgorithm::from_str("aes256gcm"),
            Some(EncryptionAlgorithm::Aes256Gcm)
        );
        assert_eq!(
            EncryptionAlgorithm::from_str("chacha20-poly1305"),
            Some(EncryptionAlgorithm::ChaCha20Poly1305)
        );
        assert_eq!(
            EncryptionAlgorithm::from_str("chacha20poly1305"),
            Some(EncryptionAlgorithm::ChaCha20Poly1305)
        );
        assert_eq!(EncryptionAlgorithm::from_str("unknown"), None);
        
        // Test conversion from algorithm to string
        assert_eq!(
            EncryptionAlgorithm::Aes128Gcm.to_string(),
            "aes128gcm"
        );
        assert_eq!(
            EncryptionAlgorithm::Aes256Gcm.to_string(),
            "aes256gcm"
        );
        assert_eq!(
            EncryptionAlgorithm::ChaCha20Poly1305.to_string(),
            "chacha20poly1305"
        );
        
        // Test key length
        assert_eq!(EncryptionAlgorithm::Aes128Gcm.key_len(), 16);
        assert_eq!(EncryptionAlgorithm::Aes256Gcm.key_len(), 32);
        assert_eq!(EncryptionAlgorithm::ChaCha20Poly1305.key_len(), 32);
    }
    
    #[test]
    fn test_key_derivation() {
        let master_key = b"master_key";
        let salt = b"salt";
        let derived_key = derive_key(master_key, salt, 10000, 16);
        
        assert_eq!(derived_key.len(), 16);
        
        // Test that different salts produce different keys
        let salt2 = b"different_salt";
        let derived_key2 = derive_key(master_key, salt2, 10000, 16);
        
        assert_ne!(derived_key, derived_key2);
        
        // Test different key lengths
        let derived_key32 = derive_key(master_key, salt, 10000, 32);
        assert_eq!(derived_key32.len(), 32);
    }
    
    #[test]
    fn test_basic_encryption_decryption() {
        let data = b"Hello, world!";
        
        // Test AES-128-GCM
        let mut enc_info = TensorEncryptionInfo::empty(
            "aes128gcm".to_string(),
            "aes128gcm".to_string(),
            "test_key".to_string(),
        );
        enc_info.enc_key = vec![0u8; 16];
        
        // Encrypt the data
        let encrypted = encrypt_data(data, &mut enc_info).unwrap();
        
        // Verify that IV and tag were updated
        assert!(!enc_info.iv.is_empty());
        assert!(!enc_info.tag.is_empty());
        
        // Decrypt the data
        let decrypted = decrypt_data(&encrypted, &enc_info).unwrap();
        
        assert_eq!(data.to_vec(), decrypted);
        
        // Test AES-256-GCM
        enc_info = TensorEncryptionInfo::empty(
            "aes128gcm".to_string(),
            "aes256gcm".to_string(),
            "test_key".to_string(),
        );
        enc_info.enc_key = vec![0u8; 32];
        
        // Encrypt the data
        let encrypted = encrypt_data(data, &mut enc_info).unwrap();
        
        // Verify that IV and tag were updated
        assert!(!enc_info.iv.is_empty());
        assert!(!enc_info.tag.is_empty());
        
        // Decrypt the data
        let decrypted = decrypt_data(&encrypted, &enc_info).unwrap();
        
        assert_eq!(data.to_vec(), decrypted);
        
        // Test ChaCha20-Poly1305
        enc_info = TensorEncryptionInfo::empty(
            "aes128gcm".to_string(),
            "chacha20poly1305".to_string(),
            "test_key".to_string(),
        );
        enc_info.enc_key = vec![0u8; 32];
        
        // Encrypt the data
        let encrypted = encrypt_data(data, &mut enc_info).unwrap();
        
        // Verify that IV and tag were updated
        assert!(!enc_info.iv.is_empty());
        assert!(!enc_info.tag.is_empty());
        
        // Decrypt the data
        let decrypted = decrypt_data(&encrypted, &enc_info).unwrap();
        
        assert_eq!(data.to_vec(), decrypted);
    }
    
    #[test]
    fn test_enc_config() {
        let master_key = vec![1u8; 32];
        let key_enc_algo = "aes256gcm".to_string();
        let data_enc_algo = "chacha20poly1305".to_string();
        let key_id = Some("test_key".to_string());
        let encrypted_tensors = vec!["tensor1".to_string(), "tensor2".to_string()];
        
        let enc_config = TensorEncryptionConfig::new(
            master_key,
            key_enc_algo,
            data_enc_algo,
            key_id,
            encrypted_tensors,
        );
        
        // Test should_encrypt_tensor
        assert!(enc_config.should_encrypt_tensor("tensor1"));
        assert!(enc_config.should_encrypt_tensor("tensor2"));
        assert!(!enc_config.should_encrypt_tensor("tensor3"));
        
        // Test generate_enc_info
        let enc_info = enc_config.generate_enc_info("tensor1").unwrap();
        
        assert_eq!(enc_info.key_enc_algo, "aes256gcm");
        assert_eq!(enc_info.data_enc_algo, "chacha20poly1305");
        assert_eq!(enc_info.key_id, "test_key");
        assert!(!enc_info.enc_key.is_empty());
        assert!(!enc_info.key_iv.is_empty());
        assert!(!enc_info.key_tag.is_empty());
        assert!(enc_info.iv.is_empty());
        assert!(enc_info.tag.is_empty());
    }
    
    #[test]
    fn test_tensor_encryption_decryption() {
        // Create a master key and encryption config
        let master_key = vec![1u8; 32];
        let enc_config = TensorEncryptionConfig::new(
            master_key,
            "aes256gcm".to_string(),
            "chacha20poly1305".to_string(),
            Some("test_key".to_string()),
            vec!["tensor1".to_string()],
        );
        
        // Generate encryption info for a tensor
        let mut enc_info = enc_config.generate_enc_info("tensor1").unwrap();
        
        // Verify that key encryption info is set
        assert_eq!(enc_info.key_enc_algo, "aes256gcm");
        assert_eq!(enc_info.data_enc_algo, "chacha20poly1305");
        assert_eq!(enc_info.key_id, "test_key");
        assert!(!enc_info.enc_key.is_empty());
        assert!(!enc_info.key_iv.is_empty());
        assert!(!enc_info.key_tag.is_empty());
        assert!(enc_info.iv.is_empty());
        assert!(enc_info.tag.is_empty());
        
        // Encrypt some tensor data
        let data = b"Tensor data";
        let encrypted = encrypt_tensor_data(data, &mut enc_info).unwrap();
        
        // Verify that data encryption info is set
        assert!(!enc_info.iv.is_empty());
        assert!(!enc_info.tag.is_empty());
        
        // Decrypt the data
        let decrypted = decrypt_tensor_data(&encrypted, &enc_info).unwrap();
        
        // Verify the decrypted data matches the original
        assert_eq!(data.to_vec(), decrypted);
    }
    
    #[test]
    fn test_key_and_data_encryption_separation() {
        // Create a master key and encryption config
        let master_key = vec![1u8; 32];
        let enc_config = TensorEncryptionConfig::new(
            master_key,
            "aes256gcm".to_string(),
            "chacha20poly1305".to_string(),
            Some("test_key".to_string()),
            vec!["tensor1".to_string()],
        );
        
        // Generate encryption info for a tensor
        let mut enc_info = enc_config.generate_enc_info("tensor1").unwrap();
        
        // Store the key encryption info
        let key_iv = enc_info.key_iv.clone();
        let key_tag = enc_info.key_tag.clone();
        let enc_key = enc_info.enc_key.clone();
        
        // Encrypt some tensor data
        let data = b"Tensor data";
        let encrypted = encrypt_tensor_data(data, &mut enc_info).unwrap();
        
        // Verify that key encryption info is preserved
        assert_eq!(enc_info.key_iv, key_iv);
        assert_eq!(enc_info.key_tag, key_tag);
        assert_eq!(enc_info.enc_key, enc_key);
        
        // Verify that data encryption info is set
        assert!(!enc_info.iv.is_empty());
        assert!(!enc_info.tag.is_empty());
        
        // Decrypt the data
        let decrypted = decrypt_tensor_data(&encrypted, &enc_info).unwrap();
        
        // Verify the decrypted data matches the original
        assert_eq!(data.to_vec(), decrypted);
    }
} 