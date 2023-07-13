pip install -r requirements.txt
curl -fsSL https://deb.nodesource.com/setup_16.x | sudo -E bash -; sudo apt-get install -y nodejs
pip install git+https://github.com/neelnanda-io/PySvelte.git

%cd ../
!bash setup.sh
%cd notebooks