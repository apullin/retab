use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let serialize_rs = manifest_dir.join("../vendor/tosa-rs/src/serialize.rs");
    println!("cargo:rerun-if-changed={}", serialize_rs.display());

    let source = fs::read_to_string(&serialize_rs).unwrap_or_else(|err| {
        panic!(
            "could not read vendored tosa-rs source at {}: {err}",
            serialize_rs.display()
        )
    });

    let old_generated_api =
        source.contains("size: t.size_()") || source.contains("size_: tensor.size");
    let patched_generated_api =
        source.contains("size: t.size()") && source.contains("size: tensor.size");

    if old_generated_api || !patched_generated_api {
        panic!(
            "vendored tosa-rs is missing the flatbuffers size-field patch; run \
             scripts/apply_vendor_patches.sh from the arm-tosa-engine repo"
        );
    }
}
