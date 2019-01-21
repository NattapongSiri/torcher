#! /bin/bash
cargo doc -p storage -p tensor --no-deps
if [ $? -eq 0 ]
then
    cargo doc --no-deps
    if [ $? -eq 0 ]
    then        
        echo -e "\033[0;32m====================================================="
        echo -e "               Build doc finish                      "
        echo -e "=====================================================\033[0m"
    else
        echo -e "\033[0;31m====================================================="
        echo -e "         Fail to build doc for root crate.           "
        echo -e "=====================================================\033[0m"
    fi
else
    echo -e "\033[0;31m====================================================="
    echo -e "Fail to build doc for either tensor or storage crate."
    echo -e "=====================================================\033[0m"
fi