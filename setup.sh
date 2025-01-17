export ORACLE_REQUEST_PIPE_PATH=/tmp/oracle_request_pipe
if ! [[ -p $ORACLE_REQUEST_PIPE_PATH ]]
then
    mkfifo $ORACLE_REQUEST_PIPE_PATH
fi
export ORACLE_RESPONSE_PIPE_PATH=/tmp/oracle_response_pipe
if ! [[ -p $ORACLE_RESPONSE_PIPE_PATH ]]
then
    mkfifo $ORACLE_RESPONSE_PIPE_PATH
fi

export FEASIBILITY_REQUEST_PIPE_PATH=/tmp/feasibility_request_pipe
if ! [[ -p $FEASIBILITY_REQUEST_PIPE_PATH ]]
then
    mkfifo $FEASIBILITY_REQUEST_PIPE_PATH
fi
export FEASIBILITY_RESPONSE_PIPE_PATH=/tmp/feasibility_response_pipe
if ! [[ -p $FEASIBILITY_RESPONSE_PIPE_PATH ]]
then
    mkfifo $FEASIBILITY_RESPONSE_PIPE_PATH
fi

export FEASIBILITY_SAMPLE_REQUEST_PIPE_PATH=/tmp/feasibility_sample_request_pipe
if ! [[ -p $FEASIBILITY_SAMPLE_REQUEST_PIPE_PATH ]]
then
    mkfifo $FEASIBILITY_SAMPLE_REQUEST_PIPE_PATH
fi
export FEASIBILITY_SAMPLE_RESPONSE_PIPE_PATH=/tmp/feasibility_sample_response_pipe
if ! [[ -p $FEASIBILITY_SAMPLE_RESPONSE_PIPE_PATH ]]
then
    mkfifo $FEASIBILITY_SAMPLE_RESPONSE_PIPE_PATH
fi

export PUSHABILITY_REQUEST_PIPE_PATH=/tmp/pushability_request_pipe
if ! [[ -p $PUSHABILITY_REQUEST_PIPE_PATH ]]
then
    mkfifo $PUSHABILITY_REQUEST_PIPE_PATH
fi
export PUSHABILITY_RESPONSE_PIPE_PATH=/tmp/pushability_response_pipe
if ! [[ -p $PUSHABILITY_RESPONSE_PIPE_PATH ]]
then
    mkfifo $PUSHABILITY_RESPONSE_PIPE_PATH
fi

export PUSHABILITY_PROJECTION_REQUEST_PIPE_PATH=/tmp/pushability_projection_request_pipe
if ! [[ -p $PUSHABILITY_PROJECTION_REQUEST_PIPE_PATH ]]
then
    mkfifo $PUSHABILITY_PROJECTION_REQUEST_PIPE_PATH
fi
export PUSHABILITY_PROJECTION_RESPONSE_PIPE_PATH=/tmp/pushability_projection_response_pipe
if ! [[ -p $PUSHABILITY_PROJECTION_RESPONSE_PIPE_PATH ]]
then
    mkfifo $PUSHABILITY_PROJECTION_RESPONSE_PIPE_PATH
fi
