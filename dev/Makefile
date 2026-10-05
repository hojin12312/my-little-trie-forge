RELEASE_VERSION := 0.1.2
.PHONY: check public-smoke package package-bottle package-check
check: public-smoke
public-smoke: install-environment all
	$(CXX) -std=c++20 -Iruntime tests/width_contract.cpp -o $(BUILD)/public-width-contract
	SPLASH_DEV_DECODE_WIDTH=8 SPLASH_DECODE_WIDTH_CAP=cost $(BUILD)/public-width-contract
	$(PYTHON) dev/tools/check_architecture.py
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py'
package: install-environment all
	$(PYTHON) dev/tools/package.py --version $(RELEASE_VERSION) --macos-min $(MACOS_MIN_VERSION)
package-bottle:
	$(PYTHON) dev/tools/package_brew.py build --version $(RELEASE_VERSION)
package-check:
	$(PYTHON) dev/tools/package_brew.py check --version $(RELEASE_VERSION)
