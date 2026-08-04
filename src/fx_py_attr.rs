use pyo3::prelude::*;

// Extraction here always reads out of a temporary `Bound` and returns the value
// past its lifetime, so the bound is `FromPyObjectOwned<'py>` — pyo3 0.27 split
// `FromPyObject` into `FromPyObject<'a, 'py>`, where `'a` borrows the source object.

pub(crate) struct AttrView<'a, 'py> {
    obj: &'a Bound<'py, PyAny>,
}

impl<'a, 'py> AttrView<'a, 'py> {
    pub(crate) fn object(&self, name: &str) -> PyResult<Bound<'py, PyAny>> {
        self.obj.getattr(name)
    }

    pub(crate) fn required_str(&self, name: &str) -> PyResult<String> {
        self.object(name)?.str()?.extract()
    }

    pub(crate) fn display(&self) -> PyResult<String> {
        self.obj.str()?.extract()
    }

    pub(crate) fn try_has(&self, name: &str) -> PyResult<bool> {
        self.obj.hasattr(name)
    }

    pub(crate) fn optional<T: FromPyObjectOwned<'py>>(&self, name: &str) -> Option<T> {
        self.obj.getattr(name).ok().and_then(|v| v.extract().ok())
    }

    pub(crate) fn call_optional<T: FromPyObjectOwned<'py>>(&self, name: &str) -> Option<T> {
        self.obj
            .call_method0(name)
            .ok()
            .and_then(|v| v.extract().ok())
    }

    pub(crate) fn or_value<T: FromPyObjectOwned<'py>>(&self, name: &str, default: T) -> T {
        self.optional(name).unwrap_or(default)
    }

    pub(crate) fn first_optional<T: FromPyObjectOwned<'py>>(&self, names: &[&str]) -> Option<T> {
        names.iter().find_map(|name| self.optional(name))
    }

    pub(crate) fn first_or<T: FromPyObjectOwned<'py>>(&self, names: &[&str], default: T) -> T {
        self.first_optional(names).unwrap_or(default)
    }

    pub(crate) fn display_optional(&self, name: &str) -> Option<String> {
        self.obj
            .getattr(name)
            .ok()
            .and_then(|v| attrs(&v).display().ok())
    }

    pub(crate) fn has(&self, name: &str) -> bool {
        self.obj.getattr(name).is_ok()
    }
}

pub(crate) fn attrs<'a, 'py>(obj: &'a Bound<'py, PyAny>) -> AttrView<'a, 'py> {
    AttrView { obj }
}

pub(crate) struct MetaView<'a, 'py> {
    dict: &'a Bound<'py, PyAny>,
}

impl<'a, 'py> MetaView<'a, 'py> {
    pub(crate) fn value(&self, key: &str) -> Option<Bound<'py, PyAny>> {
        self.dict
            .call_method1("get", (key,))
            .ok()
            .filter(|v| !v.is_none())
    }

    pub(crate) fn optional<T: FromPyObjectOwned<'py>>(&self, key: &str) -> Option<T> {
        self.value(key).and_then(|v| v.extract().ok())
    }

    pub(crate) fn display_optional(&self, key: &str) -> Option<String> {
        self.value(key).and_then(|v| attrs(&v).display().ok())
    }
}

pub(crate) fn meta<'a, 'py>(dict: &'a Bound<'py, PyAny>) -> MetaView<'a, 'py> {
    MetaView { dict }
}

pub(crate) struct ItemView<'a, 'py> {
    obj: &'a Bound<'py, PyAny>,
}

impl<'a, 'py> ItemView<'a, 'py> {
    pub(crate) fn key(&self, key: &str) -> Option<Bound<'py, PyAny>> {
        self.obj.get_item(key).ok()
    }

    pub(crate) fn optional_key<T: FromPyObjectOwned<'py>>(&self, key: &str) -> Option<T> {
        self.key(key).and_then(|v| v.extract().ok())
    }

    pub(crate) fn pairs(&self) -> Vec<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        let Ok(items) = self.obj.call_method0("items") else {
            return Vec::new();
        };
        let Ok(iter) = items.try_iter() else {
            return Vec::new();
        };
        iter.filter_map(|item| {
            let item = item.ok()?;
            let key = item.get_item(0).ok()?;
            let value = item.get_item(1).ok()?;
            Some((key, value))
        })
        .collect()
    }

    pub(crate) fn typed_key_pairs<K: FromPyObjectOwned<'py>>(&self) -> Vec<(K, Bound<'py, PyAny>)> {
        self.pairs()
            .into_iter()
            .filter_map(|(key, value)| Some((key.extract().ok()?, value)))
            .collect()
    }

    pub(crate) fn typed_pairs<K, V>(&self) -> Vec<(K, V)>
    where
        K: FromPyObjectOwned<'py>,
        V: FromPyObjectOwned<'py>,
    {
        self.pairs()
            .into_iter()
            .filter_map(|(key, value)| Some((key.extract().ok()?, value.extract().ok()?)))
            .collect()
    }
}

pub(crate) fn items<'a, 'py>(obj: &'a Bound<'py, PyAny>) -> ItemView<'a, 'py> {
    ItemView { obj }
}
