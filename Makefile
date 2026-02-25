WASM_DIR = crates/wasm-inference
DOCS_DIR = docs
PORT = 8000

build_wasm:
	rm -rf $(DOCS_DIR)/pkg
	wasm-pack build $(WASM_DIR) --target web --release
	cp -r $(WASM_DIR)/pkg $(DOCS_DIR)/

serve:
	cd $(DOCS_DIR) && python3 -m http.server $(PORT)

dev: build_wasm serve