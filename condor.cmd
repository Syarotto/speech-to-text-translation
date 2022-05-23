executable = src/evaluate.sh
getenv     = true
error      = condor.error
log        = condor.log
arguments  = /home2/tianranl/anaconda3/envs/575c/bin/python /home2/tianranl/LING575c/speech-to-text-translation
transfer_executable = false
request_memory = 6*1024
request_GPUs = 1
queue