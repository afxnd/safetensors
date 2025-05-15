# CryptoTensor 系统架构与设计模式

## 系统架构

### 文件结构

1. 核心模块
   - crypto.rs：密码学相关功能
     - 密钥管理
     - 加密算法实现
     - 完整性保护
   - tensor.rs：Tensor操作
     - 加密Tensor处理
     - 解密Tensor处理
     - 切片操作支持
   - slice.rs：切片操作
     - 加密切片处理
     - 解密切片处理
     - 内存管理

### 文件格式设计

1. 为了保持对Safetensor的兼容，在`__metadata__`下新增字段
```json
{
  "__metadata__": {
    "__encryption__": "json encoded string",
    "__policy__": "json encoded string", // 待实现
    "__signature__": "json encoded string" // 待实现
  }
}
```

2. 在`__encryption__`中保存加密的Tensor的元数据
```json
{
    "__encryption__": {
      "tensor_name1": { // 被加密的Tensor名字
        "key_enc_algo": "String", // 加密数据加密密钥时使用的算法 (aes256gcm/chacha20poly1305)
        "data_enc_algo": "String", // 加密数据时使用的算法 (aes256gcm/chacha20poly1305)
        "key_id": "String", // 主密钥ID（可选）
        "enc_key": "Vec<u8>", // 数据加密密钥密文
        "key_iv": "Vec<u8>", // 数据加密密钥密文对应的iv
        "key_tag": "Vec<u8>", // 数据加密密钥密文对应的tag
        "iv": "Vec<u8>", // 数据密文对应的iv
        "tag": "Vec<u8>" // 数据密文对应的tag
      }
    }
}
```

3. 在`__policy__`中保存受控部署策略(待实现)
```json
{
    "__policy__": {
        "version": "String", // 策略版本
        "rules": [ // 策略规则列表
            {
                "effect": "String", // allow/deny
                "conditions": [ // 条件列表
                    {
                        "type": "String", // 条件类型
                        "value": "Any" // 条件值
                    }
                ]
            }
        ],
        "metadata": { // 策略元数据
            "issuer": "String", // 签发者
            "validity": { // 有效期
                "not_before": "String",
                "not_after": "String"
            }
        }
    }
}
```

4. 在`__signature__`中保存完整性保护信息(待实现)
```json
{
    "__signature__": {
        "header": { //包含密钥管理相关字段、x.509证书相关字段, 元数据字段等
            "jku": "String", // JSON Web Key URL
            "jkw": "String", // JSON Web Key
            "kid": "String", // Key ID
            "x5u": "String", // X.509 URL
            "x5c": ["String"], // X.509 Certificate Chain
            "x5t": "String", // X.509 Certificate SHA-1 Thumbprint
            "x5t#S256": "String" // X.509 Certificate SHA-256 Thumbprint
        },
        "signature": "Vec<u8>" // 对Header进行签名产生的东西
    }
}
```

## 设计模式

### 1. 加密模式

- Tensor 级别独立加密
  - 每个 Tensor 使用独立密钥
  - 支持选择性加密
  - 密钥生命周期管理
  - 密钥轮换机制

- 密钥管理
  - 主密钥加密数据密钥
  - 尽量缩短数据加密密钥生命周期
  - 安全密钥存储

### 2. 完整性保护模式

- 文件头签名
  - 保护加密元数据
  - 保护部署策略
  - 防篡改机制
  - 证书链验证
  - 签名更新和撤销

- 身份标识
  - 模型所有者签名
  - 密钥标识管理
  - 授权验证

### 3. 性能优化模式

- 懒加载机制
  - 按需解密
  - 内存效率管理
  - 缓存策略
  - 预加载优化
  - 资源释放

- 并行处理
  - 并行解密支持
  - 分布式场景优化
  - 资源利用优化

## 组件关系

1. 核心组件
   - TensorCryptor：负责Tensor的加解密
   - CryptoManager：负责Header的解析与TensorCryptor的管理

2. 辅助组件
   - CryptoError：错误处理 