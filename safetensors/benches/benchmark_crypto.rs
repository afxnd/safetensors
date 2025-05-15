use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use safetensors::tensor::*;
use safetensors::crypto::{EncryptConfig, DecryptConfig};
use std::collections::HashMap;

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
        ("AES-128-GCM", "aes128gcm", "aes128gcm", 16),
        ("AES-256-GCM", "aes256gcm", "aes256gcm", 32),
        ("ChaCha20-Poly1305", "chacha20poly1305", "chacha20poly1305", 32),
    ];

    let ratios = [
        ("0%", 0),   // No encryption
        ("20%", 1),  // Encrypt 1 tensor
        ("40%", 2),  // Encrypt 2 tensors
        ("60%", 3),  // Encrypt 3 tensors
        ("80%", 4),  // Encrypt 4 tensors
        ("100%", 5), // Encrypt all tensors
    ];

    let mut group = c.benchmark_group("encryption_performance");
    group.measurement_time(std::time::Duration::from_secs(30));
    group.plot_config(criterion::PlotConfiguration::default()
        .summary_scale(criterion::AxisScale::Logarithmic));

    for (algo_name, key_algo, data_algo, key_size) in algorithms.iter() {
        let master_key = vec![1u8; *key_size];
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

            let enc_config = EncryptConfig::new(
                master_key.clone(),
                key_algo.to_string(),
                data_algo.to_string(),
                None,
                Some(tensors_to_encrypt),
            );

            let benchmark_id = BenchmarkId::new(
                format!("{}", algo_name),
                format!("{} encryption", ratio_name)
            );

            group.bench_with_input(benchmark_id, &metadata, |b, metadata| {
                b.iter(|| {
                    let _serialized = serialize(black_box(metadata), black_box(&None), Some(&enc_config));
                })
            });
        }
    }
    group.finish();
}

// Benchmark decryption performance for different encryption algorithms
fn bench_decryption_performance(c: &mut Criterion) {
    let (data, shape, dtype) = get_sample_data();
    let n_layers = 5;
    let algorithms = [
        ("AES-128-GCM", "aes128gcm", "aes128gcm", 16),
        ("AES-256-GCM", "aes256gcm", "aes256gcm", 32),
        ("ChaCha20-Poly1305", "chacha20poly1305", "chacha20poly1305", 32),
    ];

    let mut group = c.benchmark_group("decryption_performance");
    group.measurement_time(std::time::Duration::from_secs(30)); // target measurement time
    group.plot_config(criterion::PlotConfiguration::default()
        .summary_scale(criterion::AxisScale::Logarithmic));

    for (algo_name, key_algo, data_algo, key_size) in algorithms.iter() {
        let master_key = vec![1u8; *key_size];
        let mut metadata: HashMap<String, TensorView> = HashMap::new();
        for i in 0..n_layers {
            let tensor = TensorView::new(dtype, shape.clone(), &data[..]).unwrap();
            metadata.insert(format!("weight{i}"), tensor);
        }

        let enc_config = EncryptConfig::new(
            master_key.clone(),
            key_algo.to_string(),
            data_algo.to_string(),
            None,
            None,
        );

        let serialized = serialize(&metadata, &None, Some(&enc_config)).unwrap();
        let decrypt_config = DecryptConfig::new(master_key.clone(), None);

        group.bench_with_input(
            BenchmarkId::new("decryption", algo_name),
            &serialized,
            |b, serialized| {
                b.iter(|| {
                    let mut tensors = SafeTensors::deserialize(black_box(serialized)).unwrap();
                    tensors.load_decrypt_config(Some(&decrypt_config)).unwrap();
                })
            },
        );
    }
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
