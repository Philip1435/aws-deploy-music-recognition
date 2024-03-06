# aws-deploy-music-recognition
## Usage guidelines
To host on local machine, 
- `pip install -r requirements.txt` 
- `streamlit run app.py`
- put your own aws credentials inside .aws directory

To host on EC2
- move all of the files to an EC2 instance with `scp -i keypair.pem -r aws-deploy-music-recognition ec2-user@{your_instance_ip}:~`
- do the same run as on local machine above, then you will get your public IP