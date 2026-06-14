use pyo3::prelude::*;

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

    pub(crate) fn try_has(&self, name: &str) -> PyResult<bool> {
        self.obj.hasattr(name)
    }

    pub(crate) fn optional<T: FromPyObject<'py>>(&self, name: &str) -> Option<T> {
        self.obj.getattr(name).ok().and_then(|v| v.extract().ok())
    }

    pub(crate) fn or_value<T: FromPyObject<'py>>(&self, name: &str, default: T) -> T {
        self.optional(name).unwrap_or(default)
    }

    pub(crate) fn first_optional<T: FromPyObject<'py>>(&self, names: &[&str]) -> Option<T> {
        names.iter().find_map(|name| self.optional(name))
    }

    pub(crate) fn first_or<T: FromPyObject<'py>>(&self, names: &[&str], default: T) -> T {
        self.first_optional(names).unwrap_or(default)
    }

    pub(crate) fn display_optional(&self, name: &str) -> Option<String> {
        self.obj.getattr(name).ok().map(|v| format!("{v}"))
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

    pub(crate) fn optional<T: FromPyObject<'py>>(&self, key: &str) -> Option<T> {
        self.value(key).and_then(|v| v.extract().ok())
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

    pub(crate) fn optional_key<T: FromPyObject<'py>>(&self, key: &str) -> Option<T> {
        self.key(key).and_then(|v| v.extract().ok())
    }

    pub(crate) fn pairs(&self) -> Vec<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        let Ok(items) = self.obj.call_method0("items") else {
            return Vec::new();
        };
        let Ok(iter) = items.iter() else {
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

    pub(crate) fn typed_key_pairs<K: FromPyObject<'py>>(&self) -> Vec<(K, Bound<'py, PyAny>)> {
        self.pairs()
            .into_iter()
            .filter_map(|(key, value)| Some((key.extract().ok()?, value)))
            .collect()
    }

    pub(crate) fn typed_pairs<K, V>(&self) -> Vec<(K, V)>
    where
        K: FromPyObject<'py>,
        V: FromPyObject<'py>,
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
