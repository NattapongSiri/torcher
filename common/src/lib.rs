use std::fmt::Display;
use std::fmt;

/// Reference from https://github.com/pytorch/pytorch/blob/fb68d813be3ccbab3cbf74008a11071b9a646c75/aten/src/TH/THGeneral.h.in#L86-L89
/// THDescBuff is an array of 64 bytes of char type.
/// Since Rust support array only up to 32 elements, we need two arrays of 32 chars.
#[repr(C)]
pub struct THDescBuff {
    r#str: [u8; 64]
}

impl Display for THDescBuff {
    fn fmt(&self, f: &mut fmt::Formatter ) -> fmt::Result {
        let mut result = String::new();

        for i in 0..64 {
            if self.str[i] != 0 {
                result.push(self.str[i]  as char);
            } else {
                return write!(f, "{}", result);
            }
        }

        write!(f, "{}", result)
    }
}

/// An enum representing Storage type that compatible with Rust
/// Most notable missing type is THHalfStorage which is 16 bits float.
/// Rust support only 32/64 bits float.
pub enum StorageType {
    // float group
    THFloatStorage(&'static str),
    THDoubleStorage(&'static str),

    // byte group
    THByteStorage(&'static str),
    THCharStorage(&'static str),

    // integer group
    THShortStorage(&'static str),
    THIntStorage(&'static str),
    THLongStorage(&'static str)
}

impl StorageType {
    /// Construct the enum from string.
    pub fn from_str(def : &str) -> StorageType {
        match def {
            "THFloatStorage" | "f32" | "float" => {
                StorageType::THFloatStorage("f32")
            },
            "THDoubleStorage" | "f64" | "double" => {
                StorageType::THDoubleStorage("f64")
            },
            "THByteStorage" | "u8" | "byte" => {
                StorageType::THByteStorage("u8")
            },
            "THCharStorage" | "i8" | "char" => {
                StorageType::THCharStorage("i8")
            },
            "THShortStorage" | "i16" | "short" => {
                StorageType::THShortStorage("i16")
            },
            "THIntStorage" | "i32" | "int" => {
                StorageType::THIntStorage("i32")
            },
            "THLongStorage" | "i64" | "long" => {
                StorageType::THLongStorage("i64")
            },
            _ => {
                panic!("Unsupported storage type.")
            }
        }
    }

    /// Get a rust primitive type for current storage type.
    pub fn to_ty(&self) -> &'static str {
        match self {
            StorageType::THFloatStorage(ty) |
            StorageType::THDoubleStorage(ty) |
            StorageType::THByteStorage(ty) |
            StorageType::THCharStorage(ty) | 
            StorageType::THShortStorage(ty) |
            StorageType::THIntStorage(ty) |
            StorageType::THLongStorage(ty) => {
                ty
            },
        }
    }
}

impl Display for StorageType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            StorageType::THFloatStorage(_) => {
                write!(f, "THFloatStorage")
            },
            StorageType::THDoubleStorage(_) => {
                write!(f, "THDoubleStorage")
            },
            StorageType::THByteStorage(_) => {
                write!(f, "THByteStorage")
            },
            StorageType::THCharStorage(_) => {
                write!(f, "THCharStorage")
            },
            StorageType::THShortStorage(_) => {
                write!(f, "THShortStorage")
            },
            StorageType::THIntStorage(_) => {
                write!(f, "THIntStorage")
            },
            StorageType::THLongStorage(_) => {
                write!(f, "THLongStorage")
            },
        }
    }
}

/// An enum representing Tensor type supported by Rust.
/// Most notable missing Tensor type is 16 bits float tensor.
/// Rust support only 32/64 bits float.
pub enum TensorType {
    // float group
    THFloatTensor(&'static str),
    THDoubleTensor(&'static str),

    // byte group
    THByteTensor(&'static str),
    THCharTensor(&'static str),

    // integer group
    THShortTensor(&'static str),
    THIntTensor(&'static str),
    THLongTensor(&'static str)
}

impl TensorType {
    /// Construct the enum from string.
    pub fn from_str(def : &str) -> TensorType {
        match def {
            "THFloatTensor" | "f32" | "float" => {
                TensorType::THFloatTensor("f32")
            },
            "THDoubleTensor" | "f64" | "double" => {
                TensorType::THDoubleTensor("f64")
            },
            "THByteTensor" | "u8" | "byte" => {
                TensorType::THByteTensor("u8")
            },
            "THCharTensor" | "i8" | "char" => {
                TensorType::THCharTensor("i8")
            },
            "THShortTensor" | "i16" | "short" => {
                TensorType::THShortTensor("i16")
            },
            "THIntTensor" | "i32" | "int" => {
                TensorType::THIntTensor("i32")
            },
            "THLongTensor" | "i64" | "long" => {
                TensorType::THLongTensor("i64")
            },
            _ => {
                panic!("Unsupported tensor type.")
            }
        }
    }

    /// Get a rust primitive type for current tensor type.
    pub fn to_ty(&self) -> &'static str {
        match self {
            TensorType::THFloatTensor(ty) |
            TensorType::THDoubleTensor(ty) |
            TensorType::THByteTensor(ty) |
            TensorType::THCharTensor(ty) | 
            TensorType::THShortTensor(ty) |
            TensorType::THIntTensor(ty) |
            TensorType::THLongTensor(ty) => {
                ty
            },
        }
    }
}

impl Display for TensorType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TensorType::THFloatTensor(_) => {
                write!(f, "THFloatTensor")
            },
            TensorType::THDoubleTensor(_) => {
                write!(f, "THDoubleTensor")
            },
            TensorType::THByteTensor(_) => {
                write!(f, "THByteTensor")
            },
            TensorType::THCharTensor(_) => {
                write!(f, "THCharTensor")
            },
            TensorType::THShortTensor(_) => {
                write!(f, "THShortTensor")
            },
            TensorType::THIntTensor(_) => {
                write!(f, "THIntTensor")
            },
            TensorType::THLongTensor(_) => {
                write!(f, "THLongTensor")
            },
        }
    }
}