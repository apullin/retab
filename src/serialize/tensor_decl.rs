use std::borrow::Cow;

use super::tosa_generated::{DType, TosaTensor};

/// Serializer-local tensor declaration. Unlike `tosa_rs::TosaTensor`, constant
/// data can borrow from the TensorStore so large unchanged weights are not
/// cloned before the flatbuffer builder copies them into the final buffer.
#[derive(Debug, Clone)]
pub(super) struct SerializedTensor<'a> {
    pub(super) name: String,
    pub(super) shape: Vec<i32>,
    pub(super) dtype: DType,
    pub(super) data: Option<Cow<'a, [u8]>>,
    pub(super) variable: bool,
    pub(super) is_unranked: bool,
    pub(super) variable_name: Option<String>,
}

impl<'a> SerializedTensor<'a> {
    pub(super) fn new(name: impl Into<String>, shape: Vec<i32>, dtype: DType) -> Self {
        Self {
            name: name.into(),
            shape,
            dtype,
            data: None,
            variable: false,
            is_unranked: false,
            variable_name: None,
        }
    }

    pub(super) fn with_data(
        name: impl Into<String>,
        shape: Vec<i32>,
        dtype: DType,
        data: Cow<'a, [u8]>,
    ) -> Self {
        if data.is_empty() {
            Self::new(name, shape, dtype)
        } else {
            Self {
                name: name.into(),
                shape,
                dtype,
                data: Some(data),
                variable: false,
                is_unranked: false,
                variable_name: None,
            }
        }
    }

    pub(super) fn with_owned_data(
        name: impl Into<String>,
        shape: Vec<i32>,
        dtype: DType,
        data: Vec<u8>,
    ) -> Self {
        Self::with_data(name, shape, dtype, Cow::Owned(data))
    }
}

impl<'a> From<TosaTensor> for SerializedTensor<'a> {
    fn from(tensor: TosaTensor) -> Self {
        Self {
            name: tensor.name,
            shape: tensor.shape,
            dtype: tensor.dtype,
            data: tensor.data.map(Cow::Owned),
            variable: tensor.variable,
            is_unranked: tensor.is_unranked,
            variable_name: tensor.variable_name,
        }
    }
}
