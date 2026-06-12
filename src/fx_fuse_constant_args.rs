#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ZeroInputConstantKind {
    Full,
    FactoryCall,
}
