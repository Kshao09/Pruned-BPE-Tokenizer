# docker_runner
Run docker containers on the fly with ansible roles

# running program
Must be run with sudo privlegies
example
"sudo ruby drun.rb"

The program launches a docker container that self-provisions based on the ROLE
The ROLE value in the ruby file is the Ansible role, which will be run on the container

The docker container shares a common directory with all other docker containers, the host and a private host folder

The common directory should contain the ansible code used to self-provision the docker container
The host directory will be the same as the container name, which is randomly generated

# things to do
Have Ruby code ask for the role to be run
Enable multiple roles to be run on a single container
Create a UI for deployment
Enable easy creation of custom docker containers (templates of roles to run)
* example: a go container with vim, zsh etc
* one click deploy from a UI
