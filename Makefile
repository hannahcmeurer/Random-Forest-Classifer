
  
venv: venv/bin/activate
venv/bin/activate: requirements.txt
    test -d venv || virtualenv venv
    venv/bin/pip install -Ur requirements.txt
    touch venv/bin/activate

devbuild: venv
    venv/bin/python setup.py install

devinstall: venv
    venv/bin/pip install --editable .

test: venv
    venv/bin/pytest tests/test_vm.py -v
    
$ python RandomForestClassifier.py
