cargo doc -p storage -p tensor -p shape_derive --no-deps
@IF ERRORLEVEL 1 GOTO DepFail
cargo doc --no-deps
@IF ERRORLEVEL 1 GOTO RootFail
@echo [0;32m=====================================================
@echo                Build doc finish
@echo =====================================================[0m
@GOTO End

:DepFail
@echo [0;31m=====================================================
@echo Fail to build doc for either tensor or storage crate.
@echo =====================================================[0m
@GOTO End

:RootFail
@echo [0;31m=====================================================
@echo          Fail to build doc for root crate.
@echo =====================================================[0m

:End
