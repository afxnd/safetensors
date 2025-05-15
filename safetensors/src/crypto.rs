use crate::lib::HashMap;
use serde::{Deserialize, Serialize};
use ring::{aead, rand::{self, SecureRandom}};
use std::rc::Rc;
use zeroize::Zeroizing;
use std::fmt;
use once_cell::sync::OnceCell;

/// Supported encryption algorithms for tensor data encryption
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncryptionAlgorithm {
    /// AES-128-GCM: Advanced Encryption Standard with 128-bit key and Galois/Counter Mode
    Aes128Gcm,
    /// AES-256-GCM: Advanced Encryption Standard with 256-bit key and Galois/Counter Mode
    Aes256Gcm,
    /// ChaCha20-Poly1305: ChaCha20 stream cipher with Poly1305 message authentication code
    ChaCha20Poly1305,
}

impl EncryptionAlgorithm {
    /// Convert a string representation to an encryption algorithm
    /// 
    /// # Arguments
    /// 
    /// * `s` - The string representation of the algorithm (case-insensitive, hyphens ignored)
    /// 
    /// # Returns
    /// 
    /// * `Some(EncryptionAlgorithm)` if the string matches a supported algorithm
    /// * `None` if the string does not match any supported algorithm
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

    /// Convert the encryption algorithm to its string representation
    /// 
    /// # Returns
    /// 
    /// A string representation of the algorithm in lowercase without hyphens
    pub fn to_string(&self) -> String {
        match self {
            EncryptionAlgorithm::Aes128Gcm => "aes128gcm".to_string(),
            EncryptionAlgorithm::Aes256Gcm => "aes256gcm".to_string(),
            EncryptionAlgorithm::ChaCha20Poly1305 => "chacha20poly1305".to_string(),
        }
    }

    /// Get the appropriate AEAD algorithm from the ring crate
    /// 
    /// # Returns
    /// 
    /// A reference to the corresponding AEAD algorithm implementation
    pub fn get_aead_algo(&self) -> &'static aead::Algorithm {
        match self {
            EncryptionAlgorithm::Aes128Gcm => &aead::AES_128_GCM,
            EncryptionAlgorithm::Aes256Gcm => &aead::AES_256_GCM,
            EncryptionAlgorithm::ChaCha20Poly1305 => &aead::CHACHA20_POLY1305,
        }
    }
    
    /// Get the required key length in bytes for the algorithm
    /// 
    /// # Returns
    /// 
    /// The number of bytes required for the encryption key
    pub fn key_len(&self) -> usize {
        match self {
            EncryptionAlgorithm::Aes128Gcm => 16, // 128 bits
            EncryptionAlgorithm::Aes256Gcm => 32, // 256 bits
            EncryptionAlgorithm::ChaCha20Poly1305 => 32, // 256 bits
        }
    }

    /// Get the authentication tag length in bytes for the algorithm
    /// 
    /// # Returns
    /// 
    /// The number of bytes in the authentication tag
    pub fn tag_len(&self) -> usize {
        match self {
            EncryptionAlgorithm::Aes128Gcm => 16,
            EncryptionAlgorithm::Aes256Gcm => 16,
            EncryptionAlgorithm::ChaCha20Poly1305 => 16,
        }
    }

    /// Create an AEAD tag from raw bytes
    /// 
    /// # Arguments
    /// 
    /// * `tag_bytes` - The raw bytes to create the tag from
    /// 
    /// # Returns
    /// 
    /// * `Ok(aead::Tag)` if the bytes are valid for the algorithm
    /// * `Err(String)` if the bytes are not the correct length
    pub fn create_tag(&self, tag_bytes: &[u8]) -> Result<aead::Tag, String> {
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

impl Default for EncryptionAlgorithm {
    /// Returns the default encryption algorithm (AES-256-GCM)
    /// 
    /// This is chosen as the default because it provides the highest security level
    /// among the supported algorithms.
    fn default() -> Self {
        EncryptionAlgorithm::Aes256Gcm
    }
}

/// Error types that can occur during cryptographic operations
#[derive(Debug)]
pub enum CryptoError {
    /// Failed to unwrap (decrypt) a tensor's encryption key
    KeyUnwrap {
        /// Name of the tensor that failed to unwrap
        tensor_name: String,
        /// Source of the error
        source: String,
    },
    /// Failed to decrypt tensor data
    Decrypt {
        /// Name of the tensor that failed to decrypt
        tensor_name: String,
        /// Source of the error
        source: String,
    },
    /// Failed to access tensor data
    DataAccess {
        /// Name of the tensor that failed to access
        tensor_name: String,
    },
    /// Invalid encryption algorithm specified
    InvalidAlgorithm(String),
    /// Invalid key length for the specified algorithm
    InvalidKeyLength {
        /// Expected key length in bytes
        expected: usize,
        /// Actual key length in bytes
        got: usize,
    },
    /// Invalid authentication tag length
    InvalidTagLength {
        /// Expected tag length in bytes
        expected: usize,
        /// Actual tag length in bytes
        got: usize,
    },
    /// Invalid initialization vector (IV) length
    InvalidIv {
        /// Expected IV length in bytes
        expected: usize,
        /// Actual IV length in bytes
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
    InvalidKeyId,
}

impl fmt::Display for CryptoError {
    /// Format the error message for display
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CryptoError::KeyUnwrap { tensor_name, source } => {
                write!(f, "Failed to unwrap key for tensor {}: {}", tensor_name, source)
            }
            CryptoError::Decrypt { tensor_name, source } => {
                write!(f, "Failed to decrypt data for tensor {}: {}", tensor_name, source)
            }
            CryptoError::DataAccess { tensor_name } => {
                write!(f, "Failed to access data for tensor {}", tensor_name)
            }
            CryptoError::InvalidAlgorithm(algo) => {
                write!(f, "Invalid encryption algorithm: {}", algo)
            }
            CryptoError::InvalidKeyLength { expected, got } => {
                write!(f, "Invalid key length: expected {} bytes, got {} bytes", expected, got)
            }
            CryptoError::InvalidTagLength { expected, got } => {
                write!(f, "Invalid tag length: expected {} bytes, got {} bytes", expected, got)
            }
            CryptoError::InvalidIv { expected, got } => {
                write!(f, "Invalid initialization vector length: expected {} bytes, got {} bytes", expected, got)
            }
            CryptoError::RandomGeneration(e) => {
                write!(f, "Failed to generate random data: {}", e)
            }
            CryptoError::KeyCreation(e) => {
                write!(f, "Failed to create encryption key: {}", e)
            }
            CryptoError::Encryption(e) => {
                write!(f, "Failed to encrypt data: {}", e)
            }
            CryptoError::Decryption(e) => {
                write!(f, "Failed to decrypt data: {}", e)
            }
            CryptoError::MissingMasterKey => {
                write!(f, "Missing master key")
            }
            CryptoError::InvalidKeyId => {
                write!(f, "Invalid key ID")
            }
        }
    }
}

impl std::error::Error for CryptoError {}

impl From<CryptoError> for crate::tensor::SafeTensorError {
    /// Convert a CryptoError into a SafeTensorError
    fn from(error: CryptoError) -> Self {
        crate::tensor::SafeTensorError::CryptoError(error.to_string())
    }
}

/// Configuration for managing encryption of multiple tensors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptConfig {
    /// Master key used to encrypt individual tensor keys
    pub master_key: Vec<u8>,
    /// Algorithm used to encrypt the tensor keys
    pub key_enc_algo: String,
    /// Algorithm used to encrypt the tensor data
    pub data_enc_algo: String,
    /// Optional key identifier for key management
    pub key_id: Option<String>,
    /// Optional list of tensor names to encrypt. If None, all tensors will be encrypted
    pub tensor_names: Option<Vec<String>>,
}

impl EncryptConfig {
    /// Create a new encryption configuration
    /// 
    /// # Arguments
    /// 
    /// * `master_key` - The master key used to encrypt tensor keys
    /// * `key_enc_algo` - The algorithm used to encrypt tensor keys
    /// * `data_enc_algo` - The algorithm used to encrypt tensor data
    /// * `key_id` - Optional key identifier for key management
    /// * `tensor_names` - Optional list of tensor names to encrypt. If None, all tensors will be encrypted
    /// 
    /// # Returns
    /// 
    /// A new EncryptConfig instance
    pub fn new(
        master_key: Vec<u8>,
        key_enc_algo: String,
        data_enc_algo: String,
        key_id: Option<String>,
        tensor_names: Option<Vec<String>>,
    ) -> Self {
        Self {
            master_key,
            key_enc_algo,
            data_enc_algo,
            key_id,
            tensor_names,
        }
    }
}

/// Configuration for decrypting encrypted tensors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecryptConfig {
    /// Master key used to decrypt individual tensor keys
    pub master_key: Vec<u8>,
    /// Optional key identifier for key management
    pub key_id: Option<String>,
}

impl DecryptConfig {
    /// Create a new decryption configuration
    /// 
    /// # Arguments
    /// 
    /// * `master_key` - The master key used to decrypt tensor keys
    /// * `key_id` - Optional key identifier for key management
    /// 
    /// # Returns
    /// 
    /// A new DecryptConfig instance
    pub fn new(master_key: Vec<u8>, key_id: Option<String>) -> Self {
        Self {
            master_key,
            key_id,
        }
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
/// * `Err(CryptoError)` - If encryption fails
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
) -> Result<(Vec<u8>, Vec<u8>), CryptoError> {
    // Validate inputs
    if key.is_empty() {
        return Err(CryptoError::InvalidKeyLength {
            expected: 1, // Any non-zero value
            got: 0,
        });
    }

    let algo = EncryptionAlgorithm::from_str(algo_name)
        .ok_or_else(|| CryptoError::InvalidAlgorithm(algo_name.to_string()))?;
    
    // Validate key length
    if key.len() != algo.key_len() {
        return Err(CryptoError::InvalidKeyLength {
            expected: algo.key_len(),
            got: key.len(),
        });
    }

    // If input is empty, return empty IV and tag
    if in_out.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }

    let aead_algo = algo.get_aead_algo();
    let key = aead::UnboundKey::new(aead_algo, key)
        .map_err(|e| CryptoError::KeyCreation(e.to_string()))?;
    let key = aead::LessSafeKey::new(key);
    
    // Generate a new nonce
    let mut nonce_bytes = vec![0u8; aead_algo.nonce_len()];
    let rng = rand::SystemRandom::new();
    rng.fill(&mut nonce_bytes)
        .map_err(|e| CryptoError::RandomGeneration(e.to_string()))?;
    let nonce = aead::Nonce::assume_unique_for_key(nonce_bytes.clone().try_into().unwrap());
    
    // Encrypt the data in place
    let tag = key.seal_in_place_separate_tag(nonce, aead::Aad::empty(), in_out)
        .map_err(|e| CryptoError::Encryption(e.to_string()))?;
    
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
/// * `Err(CryptoError)` - If decryption fails
/// 
/// # Errors
/// 
/// * `InvalidKeyLength` - If the key length is invalid for the algorithm
/// * `InvalidAlgorithm` - If the algorithm name is not supported
/// * `InvalidIv` - If the IV length is invalid
/// * `InvalidTagLength` - If the tag length is invalid
/// * `KeyCreation` - If key creation fails
/// * `Decryption` - If the decryption operation fails
fn decrypt_data(
    in_out: &mut [u8],
    key: &[u8],
    algo_name: &str,
    iv: &[u8],
    tag: &[u8],
) -> Result<(), CryptoError> {
    // Validate inputs
    if key.is_empty() {
        return Err(CryptoError::InvalidKeyLength {
            expected: 1, // Any non-zero value
            got: 0,
        });
    }

    let algo = EncryptionAlgorithm::from_str(algo_name)
        .ok_or_else(|| CryptoError::InvalidAlgorithm(algo_name.to_string()))?;
    
    // Validate key length
    if key.len() != algo.key_len() {
        return Err(CryptoError::InvalidKeyLength {
            expected: algo.key_len(),
            got: key.len(),
        });
    }

    // If all inputs are empty, this is an empty data case
    if in_out.is_empty() && iv.is_empty() && tag.is_empty() {
        return Ok(());
    }

    // If any of the required components is missing, return error
    if iv.is_empty() || tag.is_empty() {
        return Err(CryptoError::InvalidIv {
            expected: 1, // Any non-zero value
            got: 0,
        });
    }
    
    let aead_algo = algo.get_aead_algo();
    let key = aead::UnboundKey::new(aead_algo, key)
        .map_err(|e| CryptoError::KeyCreation(e.to_string()))?;
    
    let key = aead::LessSafeKey::new(key);
    
    let nonce = aead::Nonce::try_assume_unique_for_key(iv)
        .map_err(|_e| CryptoError::InvalidIv {
            expected: aead_algo.nonce_len(),
            got: iv.len(),
        })?;
    
    // Create tag using algorithm-specific method
    let tag = algo.create_tag(tag)
        .map_err(|_e| CryptoError::InvalidTagLength {
            expected: algo.tag_len(),
            got: tag.len(),
        })?;
    
    // Decrypt in place using separate tag
    key.open_in_place_separate_tag(nonce, aead::Aad::empty(), tag, in_out, 0..)
        .map_err(|e| CryptoError::Decryption(e.to_string()))?;
    
    Ok(())
}

/// Information about encrypted tensor data and methods for encryption/decryption
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorCryptor<'data> {
    /// Algorithm used to encrypt the tensor key
    pub key_enc_algo: String,
    /// Algorithm used to encrypt the tensor data
    pub data_enc_algo: String,
    /// Optional key identifier for key management
    pub key_id: Option<String>,
    /// Encrypted tensor key
    pub enc_key: Vec<u8>,
    /// Initialization vector for key encryption
    pub key_iv: Vec<u8>,
    /// Authentication tag for key encryption
    pub key_tag: Vec<u8>,
    /// Initialization vector for data encryption
    pub iv: Vec<u8>,
    /// Authentication tag for data encryption
    pub tag: Vec<u8>,
    /// Buffer for decrypted data
    #[serde(skip)]
    pub buffer: OnceCell<Vec<u8>>,
    /// Master key for key encryption/decryption
    #[serde(skip)]
    master_key: Rc<[u8]>,
    /// Phantom data for lifetime tracking
    #[serde(skip)]
    _phantom: std::marker::PhantomData<&'data ()>,
}

impl<'data> TensorCryptor<'data> {
    /// Create a new TensorCryptor from encryption configuration
    /// 
    /// # Arguments
    /// 
    /// * `config` - The encryption configuration to use
    /// 
    /// # Returns
    /// 
    /// A new TensorCryptor instance initialized with the configuration
    pub fn from_config(config: &EncryptConfig) -> Self {
        Self {
            key_enc_algo: config.key_enc_algo.clone(),
            data_enc_algo: config.data_enc_algo.clone(),
            key_id: config.key_id.clone(),
            enc_key: Vec::new(),
            key_iv: Vec::new(),
            key_tag: Vec::new(),
            iv: Vec::new(),
            tag: Vec::new(),
            buffer: OnceCell::new(),
            master_key: Rc::from(config.master_key.as_slice()),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Generate a random key for data encryption
    /// 
    /// # Returns
    /// 
    /// * `Ok(Vec<u8>)` - A randomly generated key of the appropriate length
    /// * `Err(CryptoError)` - If key generation fails
    /// 
    /// # Errors
    /// 
    /// * `InvalidAlgorithm` - If the algorithm name is not supported
    /// * `RandomGeneration` - If random number generation fails
    fn random_key(&self) -> Result<Vec<u8>, CryptoError> {
        let data_algo = EncryptionAlgorithm::from_str(&self.data_enc_algo)
            .ok_or_else(|| CryptoError::InvalidAlgorithm(self.data_enc_algo.clone()))?;
        
        let mut key = vec![0u8; data_algo.key_len()];
        let rng = rand::SystemRandom::new();
        rng.fill(&mut key)
            .map_err(|e| CryptoError::RandomGeneration(e.to_string()))?;
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
    /// * `Err(CryptoError)` - If key wrapping fails
    /// 
    /// # Errors
    /// 
    /// * `Encryption` - If key encryption fails
    fn wrap_key(&mut self, key: &[u8]) -> Result<(), CryptoError> {
        let mut key_buf = key.to_vec();
        let (enc_key_iv, enc_key_tag) = encrypt_data(&mut key_buf, &self.master_key, &self.key_enc_algo)?;
        self.enc_key = key_buf;
        self.key_iv = enc_key_iv;
        self.key_tag = enc_key_tag;
        Ok(())
    }

    /// Unwrap (decrypt) a key using the master key
    /// 
    /// # Returns
    /// 
    /// * `Ok(Vec<u8>)` - The unwrapped key
    /// * `Err(CryptoError)` - If key unwrapping fails
    /// 
    /// # Errors
    /// 
    /// * `MissingMasterKey` - If the master key is not set
    /// * `Decryption` - If key decryption fails
    fn unwrap_key(&self) -> Result<Vec<u8>, CryptoError> {
        // Validate master key
        if self.master_key.is_empty() {
            return Err(CryptoError::MissingMasterKey);
        }

        let mut decrypted_key_vec = self.enc_key.clone();
        decrypt_data(
            &mut decrypted_key_vec,
            &self.master_key,
            &self.key_enc_algo,
            &self.key_iv,
            &self.key_tag,
        )?;
        Ok(decrypted_key_vec)
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
    /// * `Err(CryptoError)` - If decryption fails
    /// 
    /// # Errors
    /// 
    /// * `KeyUnwrap` - If key unwrapping fails
    /// * `Decryption` - If data decryption fails
    pub fn decrypt(&'data self, data: &[u8]) -> Result<&'data [u8], CryptoError> {
        self.buffer.get_or_try_init(|| {
            let data_key = Zeroizing::new(self.unwrap_key()?);
            
            let mut buffer = data.to_vec();
            decrypt_data(
                &mut buffer,
                data_key.as_slice(),
                &self.data_enc_algo,
                &self.iv,
                &self.tag,
            )?;

            Ok(buffer)
        }).map(|vec_ref| vec_ref.as_slice())
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
    /// * `Err(CryptoError)` - If encryption fails
    /// 
    /// # Errors
    /// 
    /// * `RandomGeneration` - If key generation fails
    /// * `Encryption` - If data encryption fails
    /// * `KeyCreation` - If key creation fails
    pub fn encrypt(&mut self, data: &[u8]) -> Result<(), CryptoError> {
        // Generate random data encryption key
        let data_key = Zeroizing::new(self.random_key()?);
        // Copy data to buffer, prepare in-place encryption
        let mut buffer = data.to_vec();
        let (iv, tag) = encrypt_data(
            &mut buffer,
            &data_key,
            &self.data_enc_algo,
        )?;
        self.iv = iv;
        self.tag = tag;
        self.wrap_key(&data_key)?;
        self.buffer.set(buffer).ok();
        Ok(())
    }
}

/// Manager for handling encryption and decryption of multiple tensors
#[derive(Debug)]
pub struct CryptoManager<'data> {
    /// Mapping from tensor names to their encryptors
    cryptors: Option<HashMap<String, TensorCryptor<'data>>>,
    /// Master key for key encryption/decryption
    master_key: Option<Rc<[u8]>>,
}

impl<'data> CryptoManager<'data> {
    /// Get the encryptor for a specific tensor
    /// 
    /// # Arguments
    /// 
    /// * `tensor_name` - The name of the tensor
    /// 
    /// # Returns
    /// 
    /// * `Some(&TensorCryptor)` - If an encryptor exists for the tensor
    /// * `None` - If no encryptor exists for the tensor
    pub fn get(&self, tensor_name: &str) -> Option<&TensorCryptor<'data>> {
        self.cryptors.as_ref()?.get(tensor_name)
    }

    /// Get a mutable reference to the encryptor for a specific tensor
    /// 
    /// # Arguments
    /// 
    /// * `tensor_name` - The name of the tensor
    /// 
    /// # Returns
    /// 
    /// * `Some(&mut TensorCryptor)` - If an encryptor exists for the tensor
    /// * `None` - If no encryptor exists for the tensor
    pub fn get_mut(&mut self, tensor_name: &str) -> Option<&mut TensorCryptor<'data>> {
        self.cryptors.as_mut()?.get_mut(tensor_name)
    }

    /// Create a new encryptor mapping from encryption configuration
    /// 
    /// # Arguments
    /// 
    /// * `config` - Optional encryption configuration
    /// * `tensor_names` - List of all available tensor names
    /// 
    /// # Returns
    /// 
    /// A new CryptoManager instance. If no configuration is provided or no tensors
    /// are selected for encryption, the manager will be initialized without any
    /// encryptors.
    pub fn from_encrypt_config(config: Option<&EncryptConfig>, tensor_names: Vec<String>) -> Self {
        match config {
            Some(config) => {
                // If tensor_names is None in the config, encrypt all tensors
                let tensors_to_encrypt = match &config.tensor_names {
                    None => tensor_names,
                    Some(names) => {
                        // Otherwise, only encrypt the tensors specified in the config that exist in tensor_names
                        names.iter()
                            .filter(|name| tensor_names.contains(name))
                            .cloned()
                            .collect()
                    }
                };
                
                // Create the crypto manager
                if tensors_to_encrypt.is_empty() {
                    Self {
                        cryptors: None,
                        master_key: None,
                    }
                } else {
                    Self {
                        cryptors: Some(tensors_to_encrypt.iter()
                            .map(|name| {
                                let cryptor = TensorCryptor::from_config(config);
                                (name.clone(), cryptor)
                            })
                            .collect()),
                        master_key: Some(Rc::from(&config.master_key[..])),
                    }
                }
            },
            None => Self {
                cryptors: None,
                master_key: None,
            }
        }
    }

    /// Set the master key for decryption from the given DecryptConfig
    /// 
    /// This will update both the CryptoManager's master key and all TensorCryptor instances.
    /// 
    /// # Arguments
    /// 
    /// * `config` - Optional decryption configuration
    /// 
    /// # Returns
    /// 
    /// * `Ok(())` - If the master key was successfully set
    /// * `Err(CryptoError)` - If key ID validation fails
    /// 
    /// # Errors
    /// 
    /// * `InvalidKeyId` - If the key ID in the config doesn't match the encryptor's key ID
    pub fn from_decrypt_config(&mut self, config: Option<&DecryptConfig>) -> Result<(), CryptoError> {
        match config {
            Some(config) => {
                // Create a new Rc for the master key
                let master_key: Rc<[u8]> = Rc::from(&config.master_key[..]);
                // Update the CryptoManager's master key
                self.master_key = Some(master_key.clone());
                // Update all TensorCryptor instances with the new master key
                if let Some(cryptors) = &mut self.cryptors {
                    for cryptor in cryptors.values_mut() {
                        // Validate key_id
                        match (cryptor.key_id.as_ref(), config.key_id.as_ref()) {
                            (None, None) => (), // Both are None, valid
                            (Some(id1), Some(id2)) if id1 == id2 => (), // Both match, valid
                            _ => return Err(CryptoError::InvalidKeyId), // Invalid key_id
                        }
                        cryptor.master_key = master_key.clone();
                    }
                }
            }
            None => {
                self.master_key = None;
            }
        }
        Ok(())
    }

    /// Generate the metadata string for serialization
    /// 
    /// # Returns
    /// 
    /// * `Ok(Some(HashMap))` - If there are encryptors to serialize
    /// * `Ok(None)` - If there are no encryptors to serialize
    /// * `Err(CryptoError)` - If serialization fails
    /// 
    /// # Errors
    /// 
    /// * `Encryption` - If JSON serialization fails
    pub fn to_string(&self) -> Result<Option<HashMap<String, String>>, CryptoError> {
        if let Some(cryptors) = &self.cryptors {
            match serde_json::to_string(cryptors) {
                Ok(json) => Ok(Some(HashMap::from([("__encryption__".to_string(), json)]))),
                Err(e) => Err(CryptoError::Encryption(e.to_string())),
            }
        } else {
            Ok(None)
        }
    }

    /// Create a new encryptor mapping from metadata
    /// 
    /// # Arguments
    /// 
    /// * `metadata` - Optional metadata containing encryption information
    /// 
    /// # Returns
    /// 
    /// * `Some(CryptoManager)` - If valid encryption metadata was found
    /// * `None` - If no valid encryption metadata was found
    pub fn from_metadata(metadata: &Option<HashMap<String, String>>) -> Option<Self> {
        // Get encryption info from metadata
        let encryption_info = metadata.as_ref()?.get("__encryption__")?.to_string();        
        // Parse encryption info directly into TensorCryptor map
        let crypto_info: HashMap<String, TensorCryptor<'data>> = serde_json::from_str(&encryption_info).ok()?;
        
        Some(Self {
            cryptors: Some(crypto_info),
            master_key: None,
        })
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
    /// * `Err(CryptoError)` - If decryption fails
    pub fn silent_decrypt(&'data self, tensor_name: &str, data: &'data [u8]) -> Result<&'data [u8], CryptoError> {
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
    /// * `Err(CryptoError)` - If encryption fails
    pub fn silent_encrypt(&mut self, tensor_name: &str, data: &[u8]) -> Result<(), CryptoError> {
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
    use std::collections::HashMap;

    /// Test encryption and decryption of empty tensor data
    #[test]
    fn test_tensor_cryptor_empty_data() {
        let master_key = vec![1u8; 32];
        let config = EncryptConfig::new(
            master_key.clone(),
            "aes256gcm".to_string(),
            "aes256gcm".to_string(),
            None,
            None,
        );

        let mut encryptor = TensorCryptor::from_config(&config);
        let empty_data = b"";
        assert!(encryptor.encrypt(empty_data).is_ok());
        let encrypted_empty = encryptor.buffer.get().unwrap();
        assert!(encrypted_empty.is_empty()); // Should be empty encrypted data

        let mut decryptor = TensorCryptor::from_config(&config);
        decryptor.enc_key = encryptor.enc_key.clone();
        decryptor.key_iv = encryptor.key_iv.clone();
        decryptor.key_tag = encryptor.key_tag.clone();
        decryptor.iv = encryptor.iv.clone();
        decryptor.tag = encryptor.tag.clone();
        let decrypted_empty = decryptor.decrypt(encrypted_empty).unwrap();
        assert_eq!(decrypted_empty, empty_data);
    }

    /// Test encryption and decryption of non-empty tensor data
    #[test]
    fn test_tensor_cryptor_non_empty_data() {
        let master_key = vec![1u8; 32];
        let config = EncryptConfig::new(
            master_key.clone(),
            "aes256gcm".to_string(),
            "aes256gcm".to_string(),
            None,
            None,
        );

        let mut encryptor = TensorCryptor::from_config(&config);
        let test_data = b"Hello";
        assert!(encryptor.encrypt(test_data).is_ok());
        let encrypted_data = encryptor.buffer.get().unwrap();
        assert_ne!(encrypted_data, test_data);

        let mut decryptor = TensorCryptor::from_config(&config);
        decryptor.enc_key = encryptor.enc_key.clone();
        decryptor.key_iv = encryptor.key_iv.clone();
        decryptor.key_tag = encryptor.key_tag.clone();
        decryptor.iv = encryptor.iv.clone();
        decryptor.tag = encryptor.tag.clone();
        let decrypted_data = decryptor.decrypt(encrypted_data).unwrap();
        assert_eq!(decrypted_data, test_data);
    }

    /// Test encryption and decryption of large tensor data
    #[test]
    fn test_tensor_cryptor_large_data() {
        let master_key = vec![1u8; 32];
        let config = EncryptConfig::new(
            master_key.clone(),
            "aes256gcm".to_string(),
            "aes256gcm".to_string(),
            None,
            None,
        );

        let mut encryptor = TensorCryptor::from_config(&config);
        let large_data = vec![1u8; 1024];
        assert!(encryptor.encrypt(&large_data).is_ok());
        let encrypted_data = encryptor.buffer.get().unwrap();
        assert_ne!(encrypted_data, &large_data);

        let mut decryptor = TensorCryptor::from_config(&config);
        decryptor.enc_key = encryptor.enc_key.clone();
        decryptor.key_iv = encryptor.key_iv.clone();
        decryptor.key_tag = encryptor.key_tag.clone();
        decryptor.iv = encryptor.iv.clone();
        decryptor.tag = encryptor.tag.clone();
        let decrypted_data = decryptor.decrypt(encrypted_data).unwrap();
        assert_eq!(decrypted_data, &large_data);
    }

    /// Test encryption and decryption with different algorithms for key and data
    #[test]
    fn test_tensor_cryptor_mixed_algorithms() {
        let master_key = vec![1u8; 32];
        let config = EncryptConfig::new(
            master_key.clone(),
            "aes256gcm".to_string(),
            "chacha20poly1305".to_string(),
            None,
            None,
        );

        let mut encryptor = TensorCryptor::from_config(&config);
        let test_data = b"Test data with mixed algorithms";
        assert!(encryptor.encrypt(test_data).is_ok());
        let encrypted_data = encryptor.buffer.get().unwrap();
        assert_ne!(encrypted_data, test_data);

        let mut decryptor = TensorCryptor::from_config(&config);
        decryptor.enc_key = encryptor.enc_key.clone();
        decryptor.key_iv = encryptor.key_iv.clone();
        decryptor.key_tag = encryptor.key_tag.clone();
        decryptor.iv = encryptor.iv.clone();
        decryptor.tag = encryptor.tag.clone();
        let decrypted_data = decryptor.decrypt(encrypted_data).unwrap();
        assert_eq!(decrypted_data, test_data);
    }

    /// Test encryption and decryption with all supported algorithms
    #[test]
    fn test_tensor_cryptor_all_algorithms() {
        let test_data = b"Test data for all algorithms";
        let algorithms = [
            ("aes128gcm", "aes128gcm", 16),  // AES-128-GCM uses 16 bytes key
            ("aes256gcm", "aes256gcm", 32),  // AES-256-GCM uses 32 bytes key
            ("chacha20poly1305", "chacha20poly1305", 32),  // ChaCha20-Poly1305 uses 32 bytes key
        ];

        for (key_algo, data_algo, key_len) in algorithms.iter() {
            let master_key = vec![1u8; *key_len];
            let config = EncryptConfig::new(
                master_key.clone(),
                key_algo.to_string(),
                data_algo.to_string(),
                None,
                None,
            );

            let mut encryptor = TensorCryptor::from_config(&config);
            assert!(encryptor.encrypt(test_data).is_ok());
            let encrypted_data = encryptor.buffer.get().unwrap();
            assert_ne!(encrypted_data, test_data);

            let mut decryptor = TensorCryptor::from_config(&config);
            decryptor.enc_key = encryptor.enc_key.clone();
            decryptor.key_iv = encryptor.key_iv.clone();
            decryptor.key_tag = encryptor.key_tag.clone();
            decryptor.iv = encryptor.iv.clone();
            decryptor.tag = encryptor.tag.clone();
            let decrypted_data = decryptor.decrypt(encrypted_data).unwrap();
            assert_eq!(decrypted_data, test_data);
        }
    }

    /// Test the complete workflow of encryption and decryption
    #[test]
    fn test_crypto_manager_workflow() {
        let master_key = vec![1u8; 32];
        let config = EncryptConfig::new(
            master_key.clone(),
            "aes256gcm".to_string(),
            "aes256gcm".to_string(),
            None,
            Some(vec!["tensor1".to_string()]),
        );

        let tensor_names = vec!["tensor1".to_string()];
        let mut encrypt_manager = CryptoManager::from_encrypt_config(Some(&config), tensor_names);

        // Test data encryption
        let test_data = b"Test data";
        assert!(encrypt_manager.silent_encrypt("tensor1", test_data).is_ok());
        let encrypted_data = encrypt_manager.get_encrypted_data("tensor1").unwrap();
        assert_ne!(encrypted_data, test_data);

        // Serialize encryption metadata
        let metadata = encrypt_manager.to_string().unwrap().unwrap();
        assert!(metadata.contains_key("__encryption__"));

        // Create decrypt manager from metadata
        let mut decrypt_manager = CryptoManager::from_metadata(&Some(metadata)).unwrap();
        
        // Set master key for decryption
        let decrypt_config = DecryptConfig::new(master_key.clone(), None);
        assert!(decrypt_manager.from_decrypt_config(Some(&decrypt_config)).is_ok());

        // Test decryption
        let decrypted_data = decrypt_manager.silent_decrypt("tensor1", encrypted_data).unwrap();
        assert_eq!(decrypted_data, test_data);

        // Test with empty metadata
        assert!(CryptoManager::from_metadata(&None).is_none());
        assert!(CryptoManager::from_metadata(&Some(HashMap::new())).is_none());
    }

    /// Test key_id validation in crypto manager workflow
    #[test]
    fn test_crypto_manager_key_id_validation() {
        let master_key = vec![1u8; 32];
        let key_id = Some("test-key-123".to_string());
        let config = EncryptConfig::new(
            master_key.clone(),
            "aes256gcm".to_string(),
            "aes256gcm".to_string(),
            key_id.clone(),
            Some(vec!["tensor1".to_string()]),
        );

        let tensor_names = vec!["tensor1".to_string()];
        let mut encrypt_manager = CryptoManager::from_encrypt_config(Some(&config), tensor_names);

        // Test data encryption
        let test_data = b"Test data";
        assert!(encrypt_manager.silent_encrypt("tensor1", test_data).is_ok());
        let encrypted_data = encrypt_manager.get_encrypted_data("tensor1").unwrap();
        assert_ne!(encrypted_data, test_data);

        // Serialize encryption metadata
        let metadata = encrypt_manager.to_string().unwrap().unwrap();
        assert!(metadata.contains_key("__encryption__"));

        // Create decrypt manager from metadata
        let mut decrypt_manager = CryptoManager::from_metadata(&Some(metadata)).unwrap();
        
        // Test with matching key_id
        let decrypt_config = DecryptConfig::new(master_key.clone(), key_id.clone());
        assert!(decrypt_manager.from_decrypt_config(Some(&decrypt_config)).is_ok());

        // Test with different key_id
        let different_decrypt_config = DecryptConfig::new(
            master_key.clone(),
            Some("different-key".to_string()),
        );
        assert!(decrypt_manager.from_decrypt_config(Some(&different_decrypt_config)).is_err());

        // Test with None key_id
        let none_key_id_config = DecryptConfig::new(master_key.clone(), None);
        assert!(decrypt_manager.from_decrypt_config(Some(&none_key_id_config)).is_err());
    }
}