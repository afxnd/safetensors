use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use safetensors::tensor::*;
use safetensors::crypto::{KeyMaterial, SerializeCryptoConfig, LoadPolicy};
use std::collections::HashMap;
use std::fs;

// Returns a sample data of size 2_MB
fn get_sample_data() -> (Vec<u8>, Vec<usize>, Dtype) {
    let shape = vec![1000, 500];
    let dtype = Dtype::F32;
    let n: usize = shape.iter().product::<usize>() * dtype.size(); // 4
    let data = vec![0; n];

    (data, shape, dtype)
}

// Benchmark performance of different encryption algorithms with varying encryption ratios
fn bench_encryption_performance(c: &mut Criterion) {
    let (data, shape, dtype) = get_sample_data();
    let n_layers = 5;
    let algorithms = [
        ("AES-128-GCM", "aes128gcm"),
        // ("AES-256-GCM", "aes256gcm"),
        // ("ChaCha20-Poly1305", "chacha20poly1305"),
    ];

    let ratios = [
        ("0%", 0),   // No encryption
        ("20%", 1),  // Encrypt 1 tensor
        // ("40%", 2),  // Encrypt 2 tensors
        // ("60%", 3),  // Encrypt 3 tensors
        // ("80%", 4),  // Encrypt 4 tensors
        // ("100%", 5), // Encrypt all tensors
    ];

    let mut group = c.benchmark_group("Serialize 10_MB CryptoTensors");
    group.measurement_time(std::time::Duration::from_secs(30));
    group.plot_config(criterion::PlotConfiguration::default()
        .summary_scale(criterion::AxisScale::Logarithmic));

    // Create a temporary directory for JWK files
    let temp_dir = std::env::temp_dir().join("safetensors_benchmark_jwk");
    fs::create_dir_all(&temp_dir).unwrap();

    for (algo_name, key_algo) in algorithms.iter() {
        let jwk_path = temp_dir.join(format!("{}.jwk", algo_name));
        let jku = format!("file://{}", jwk_path.to_str().unwrap());

        let enc_key = KeyMaterial::new_enc_key(
            None,
            Some(key_algo.to_string()),
            Some(format!("{}_enc", algo_name)),
            Some(jku.clone()),
        ).unwrap();
        let sign_key = KeyMaterial::new_sign_key(
            None,
            None,
            None,
            Some(format!("{}_sig", algo_name)),
            Some(jku.clone()),
        ).unwrap();

        // Write JWK file
        let jwk_content = format!(
            r#"{{
                "keys": [
                    {},
                    {}
                ]
            }}"#,
            enc_key.to_jwk().unwrap(),
            sign_key.to_jwk().unwrap()
        );
        fs::write(&jwk_path, jwk_content).unwrap();

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

            let dummy_policy = LoadPolicy::new(None, None);
            let crypto_config = SerializeCryptoConfig::new(
                "1".to_string(),
                Some(tensors_to_encrypt),
                enc_key.clone(),
                sign_key.clone(),
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
        ("AES-128-GCM", "aes128gcm"),
        ("AES-256-GCM", "aes256gcm"),
        ("ChaCha20-Poly1305", "chacha20poly1305"),
    ];

    let mut group = c.benchmark_group("Deserialize 10_MB CryptoTensors");
    group.measurement_time(std::time::Duration::from_secs(30));
    group.plot_config(criterion::PlotConfiguration::default()
        .summary_scale(criterion::AxisScale::Logarithmic));

    // Create a temporary directory for JWK files
    let temp_dir = std::env::temp_dir().join("safetensors_benchmark_jwk");
    fs::create_dir_all(&temp_dir).unwrap();

    for (algo_name, key_algo) in algorithms.iter() {
        let jwk_path = temp_dir.join(format!("{}.jwk", algo_name));
        let jku = format!("file://{}", jwk_path.to_str().unwrap());

        let enc_key = KeyMaterial::new_enc_key(
            None,
            Some(key_algo.to_string()),
            Some(format!("{}_enc", algo_name)),
            Some(jku.clone()),
        ).unwrap();
        let sign_key = KeyMaterial::new_sign_key(
            None,
            None,
            None,
            Some(format!("{}_sig", algo_name)),
            Some(jku.clone()),
        ).unwrap();

        // Write JWK file
        let jwk_content = format!(
            r#"{{
                "keys": [
                    {},
                    {}
                ]
            }}"#,
            enc_key.to_jwk().unwrap(),
            sign_key.to_jwk().unwrap()
        );
        fs::write(&jwk_path, jwk_content).unwrap();

        let mut metadata: HashMap<String, TensorView> = HashMap::new();
        for i in 0..n_layers {
            let tensor = TensorView::new(dtype, shape.clone(), &data[..]).unwrap();
            metadata.insert(format!("weight{i}"), tensor);
        }

        let dummy_policy = LoadPolicy::new(None, None);
        let crypto_config = SerializeCryptoConfig::new(
            "1".to_string(),
            None,
            enc_key.clone(),
            sign_key.clone(),
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
