@echo off

:: <name> <activation> <units> <window size> <batch size> <epoch count> <# of tests>
:: Default parameters in order:
:: python experiments.py 'unknown' 'n' 16 50 25 2 1

:: Dataset Name ::
set datset=mkygls

set /A batch=25
set /A epochs=100
set /A tests=1

:: Activation Functions ::
set acts[0]=n
set acts[1]=m1
set acts[2]=m2
set acts[3]=m3
set acts[4]=m4
set acts[5]=m5
set acts[6]=m6
set acts[7]=m7

:: names as an array ::
set nme=dft_test

:: Units Sizes ::
set /A unts[0]=8
set /A unts[1]=16
set /A unts[2]=24

:: Activate venv ::
call .\Scripts\activate

:: Loop Variables ::
set /A count=0
set /A max_count=4

:: Main Loop ::
:again
echo Test: %count%

:: INPUT SIZE 32 ::

set /A size=32

:: Clear all previous weight matrices ::
start python experiments.py %unts[0]%
start python experiments.py %unts[1]%
start python experiments.py %unts[2]%
timeout /t 5

:: Config m2 ::
start /wait python experiments.py %nme% %acts[2]% %unts[0]% %size% %batch% %epochs% %tests% %datset%
start /wait python experiments.py %nme% %acts[2]% %unts[1]% %size% %batch% %epochs% %tests% %datset%
start /wait python experiments.py %nme% %acts[2]% %unts[2]% %size% %batch% %epochs% %tests% %datset%

:: INPUT SIZE 64 ::

set /A size=64

:: Clear all previous weight matrices ::
start python experiments.py %unts[0]%
start python experiments.py %unts[1]%
start python experiments.py %unts[2]%
timeout /t 5

:: Config m2 ::
start /wait python experiments.py %nme% %acts[2]% %unts[0]% %size% %batch% %epochs% %tests% %datset%
start /wait python experiments.py %nme% %acts[2]% %unts[1]% %size% %batch% %epochs% %tests% %datset%
start /wait python experiments.py %nme% %acts[2]% %unts[2]% %size% %batch% %epochs% %tests% %datset%

set /A count=count+1

:: Loop Check ::
if %count%==%max_count% (goto end) else (goto again)

:end
echo "Ending Tests ... "
exit /B 0