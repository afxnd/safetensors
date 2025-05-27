use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use safetensors::tensor::*;
use safetensors::crypto::{KeyMaterial, SerializeCryptoConfig, LoadPolicy};
use std::collections::HashMap;
use ring::signature::{Ed25519KeyPair, KeyPair};
use ring::rand::SystemRandom;
use std::fs;
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};

// Returns a sample data of size 2_MB
fn get_sample_data() -> (Vec<u8>, Vec<usize>, Dtype) {
    let shape = vec![1000, 500];
    let dtype = Dtype::F32;
    let n: usize = shape.iter().product::<usize>() * dtype.size(); // 4
    let data = vec![0; n];

    (data, shape, dtype)
}

// Generate Ed25519 key pair
fn generate_ed25519_keypair() -> (Vec<u8>, Vec<u8>) {
    let rng = SystemRandom::new();
    let pkcs8_bytes = Ed25519KeyPair::generate_pkcs8(&rng).unwrap();
    let key_pair = Ed25519KeyPair::from_pkcs8(pkcs8_bytes.as_ref()).unwrap();
    let public_key = key_pair.public_key().as_ref().to_vec();
    let private_key = pkcs8_bytes.as_ref().to_vec();
    (public_key, private_key)
}

// Generate JWK file for a given key pair
fn generate_jwk_file(enc_key: &[u8], sign_pub_key: &[u8], sign_priv_key: &[u8], kid: &str, enc_alg: &str) -> String {
    let jwk = format!(
        r#"{{
            "keys": [
                {{
                    "kty": "oct",
                    "alg": "{}",
                    "k": "{}",
                    "kid": "{}_enc"
                }},
                {{
                    "kty": "okp",
                    "alg": "Ed25519",
                    "x": "{}",
                    "d": "{}",
                    "kid": "{}_sig"
                }}
            ]
        }}"#,
        enc_alg,
        BASE64.encode(enc_key),
        kid,
        BASE64.encode(sign_pub_key),
        BASE64.encode(sign_priv_key),
        kid
    );
    jwk
}

// Benchmark performance of different encryption algorithms with varying encryption ratios
fn bench_encryption_performance(c: &mut Criterion) {
    let (data, shape, dtype) = get_sample_data();
    let n_layers = 5;
    let algorithms = [
        ("AES-128-GCM", "aes128gcm", 16),
        ("AES-256-GCM", "aes256gcm", 32),
        ("ChaCha20-Poly1305", "chacha20poly1305", 32),
    ];

    let ratios = [
        ("0%", 0),   // No encryption
        ("20%", 1),  // Encrypt 1 tensor
        ("40%", 2),  // Encrypt 2 tensors
        ("60%", 3),  // Encrypt 3 tensors
        ("80%", 4),  // Encrypt 4 tensors
        ("100%", 5), // Encrypt all tensors
    ];

    let mut group = c.benchmark_group("Serialize 10_MB CryptoTensor");
    group.measurement_time(std::time::Duration::from_secs(30));
    group.plot_config(criterion::PlotConfiguration::default()
        .summary_scale(criterion::AxisScale::Logarithmic));

    // Create a temporary directory for JWK files
    let temp_dir = std::env::temp_dir().join("safetensors_benchmark_jwk");
    fs::create_dir_all(&temp_dir).unwrap();

    for (algo_name, key_algo, key_size) in algorithms.iter() {
        let master_key = vec![1u8; *key_size];
        let (sign_pub_key, sign_priv_key) = generate_ed25519_keypair();
        
        // Generate JWK file for this algorithm
        let jwk_content = generate_jwk_file(&master_key, &sign_pub_key, &sign_priv_key, algo_name, key_algo);
        let jwk_path = temp_dir.join(format!("{}.jwk", algo_name));
        fs::write(&jwk_path, jwk_content).unwrap();
        let jku = format!("file://{}", jwk_path.to_str().unwrap());

        for (ratio_name, n_encrypted) in ratios.iter() {
            let mut metadata: HashMap<String, TensorView> = HashMap::new();
            for i in 0..n_layers {
                let tensor = TensorView::new(dtype, shape.clone(), &data[..]).unwrap();
                metadata.insert(format!("weight{i}"), tensor);
            }

            // Select tensors to encrypt
            let tensors_to_encrypt: Vec<String> = (0..*n_encrypted)
                .map(|i| format!("weight{i}"))
                .collect();

            let enc_key = KeyMaterial::new(
                "oct".to_string(),
                key_algo.to_string(),
                Some(format!("{}_enc", algo_name)),
                Some(jku.clone()),
                Some(master_key.clone()),
                None,
                None,
            ).unwrap();

            let sign_key = KeyMaterial::new(
                "okp".to_string(),
                "Ed25519".to_string(),
                Some(format!("{}_sig", algo_name)),
                Some(jku.clone()),
                None,
                Some(sign_pub_key.clone()),
                Some(sign_priv_key.clone()),
            ).unwrap();

            let dummy_policy = LoadPolicy::new(None, None);
            let crypto_config = SerializeCryptoConfig::new(
                "1".to_string(),
                Some(tensors_to_encrypt),
                enc_key,
                sign_key,
                dummy_policy,
            ).unwrap();

            let benchmark_id = BenchmarkId::new(
                format!("{}", algo_name),
                format!("{} encryption", ratio_name)
            );

            group.bench_with_input(benchmark_id, &metadata, |b, metadata| {
                b.iter(|| {
                    let _serialized = serialize(black_box(metadata), black_box(&None), Some(&crypto_config));
                })
            });
        }
    }

    // Clean up temporary directory
    fs::remove_dir_all(temp_dir).unwrap();
    group.finish();
}

// Benchmark decryption performance for different encryption algorithms
fn bench_decryption_performance(c: &mut Criterion) {
    let (data, shape, dtype) = get_sample_data();
    let n_layers = 5;
    let algorithms = [
        ("AES-128-GCM", "aes128gcm", 16),
        ("AES-256-GCM", "aes256gcm", 32),
        ("ChaCha20-Poly1305", "chacha20poly1305", 32),
    ];

    let mut group = c.benchmark_group("Deserialize 10_MB CryptoTensor");
    group.measurement_time(std::time::Duration::from_secs(30));
    group.plot_config(criterion::PlotConfiguration::default()
        .summary_scale(criterion::AxisScale::Logarithmic));

    // Create a temporary directory for JWK files
    let temp_dir = std::env::temp_dir().join("safetensors_benchmark_jwk");
    fs::create_dir_all(&temp_dir).unwrap();

    for (algo_name, key_algo, key_size) in algorithms.iter() {
        let master_key = vec![1u8; *key_size];
        let (sign_pub_key, sign_priv_key) = generate_ed25519_keypair();
        
        // Generate JWK file for this algorithm
        let jwk_content = generate_jwk_file(&master_key, &sign_pub_key, &sign_priv_key, algo_name, key_algo);
        let jwk_path = temp_dir.join(format!("{}.jwk", algo_name));
        fs::write(&jwk_path, jwk_content).unwrap();
        let jku = format!("file://{}", jwk_path.to_str().unwrap());

        let mut metadata: HashMap<String, TensorView> = HashMap::new();
        for i in 0..n_layers {
            let tensor = TensorView::new(dtype, shape.clone(), &data[..]).unwrap();
            metadata.insert(format!("weight{i}"), tensor);
        }

        let enc_key = KeyMaterial::new(
            "oct".to_string(),
            key_algo.to_string(),
            Some(format!("{}_enc", algo_name)),
            Some(jku.clone()),
            Some(master_key.clone()),
            None,
            None,
        ).unwrap();

        let sign_key = KeyMaterial::new(
            "okp".to_string(),
            "Ed25519".to_string(),
            Some(format!("{}_sig", algo_name)),
            Some(jku.clone()),
            None,
            Some(sign_pub_key.clone()),
            Some(sign_priv_key.clone()),
        ).unwrap();

        let dummy_policy = LoadPolicy::new(None, None);
        let crypto_config = SerializeCryptoConfig::new(
            "1".to_string(),
            None,
            enc_key,
            sign_key,
            dummy_policy,
        ).unwrap();

        let serialized = serialize(&metadata, &None, Some(&crypto_config)).unwrap();

        group.bench_with_input(
            BenchmarkId::new("decryption", algo_name),
            &serialized,
            |b, serialized| {
                b.iter(|| {
                    let _tensors = SafeTensors::deserialize(black_box(serialized)).unwrap();
                })
            },
        );
    }

    // Clean up temporary directory
    fs::remove_dir_all(temp_dir).unwrap();
    group.finish();
}

// Define separate criterion groups for encryption and decryption benchmarks
criterion_group!(
    encryption_benches,
    bench_encryption_performance
);

criterion_group!(
    decryption_benches,
    bench_decryption_performance
);

// Run both benchmark groups
criterion_main!(encryption_benches, decryption_benches);
