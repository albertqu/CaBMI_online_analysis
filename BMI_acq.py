__author__ = 'Nuria', 'Albert Qu'

import numpy as np
import os, time
from arduino_delegate import ArduinoDelegate
from utils import DCache
import SETTINGS


def set_up_bmi(cnm, iE1, iE2, T1, std_filter_thres=2, debug=False, sim=False):
    """# Variables to define before running the code / designate in interface
    while True:
        try:
            E1c = input("Choose the index of the neurons to be E1 (Droppers), format: A, B")
            res1 = E1c.split(',')
            E1 = [int(res1[0]), int(res1[1])]
            break
        except:
            continue
    while True:
        try:
            E2c = input("Choose the index of the neurons to be E2 (Risers), format: C, D")
            res2 = E2c.split(',')
            E2 = [int(res2[0]), int(res2[1])]
            break
        except:
            continue

    T1 = baselineSimulation()  # value obtained by the baseline to get 30# of correct trials."""

    # %%**********************************************************
    # ******************  PARAMETERS  ****************************
    # ************************************************************
    # parameters for the function that do not change (and we may need in each
    # iteration.
    rois = np.hstack((iE1, iE2))
    units = len(rois)
    E1, E2 = np.arange(len(iE1)), np.arange(len(iE1), units)
    ens_thresh = 0.95  # essentially top fraction (#) of what each cell can contribute to the BMI
    # TODO: MT: MIGHT STILL NEED LATER
    motionThresh = 10  # value to define maximum movement to flag a motion-relaxation time
    relaxationTime = 4  # there can't be another hit in this many sec TODO: CHECK THIS ?
    motionRelaxationFrames = 5  # number of frames that there will not be due to signficant motion
    durationTrial = 30  # maximum time that mice have for a trial
    movingAverage = 1  # Moving average of frames to calculate BMI in sec
    timeout = 5  # seconds of timeout if no hit in duration trial (sec)
    expectedLengthExperiment = cnm.Ts - cnm.base  # frames that will last the online experiment (less than actual exp)
    baseLength = 2 * 60  # to establish BL todo: is this the baseline? 2 mins?
    freqMax = 18000
    freqMin = 2000
    freqmed = (freqMax - freqMin) / 2  # Calculates the mean frequency
    if std_filter_thres:
        std_ftr = DCache(20, std_filter_thres)
    if debug:
        cnm.raw_sig = np.zeros((units, cnm.Ts-cnm.base))
        cnm.base_vals = np.zeros((units, cnm.Ts-cnm.base))

    # values of parameters in frames
    frameRate = cnm.params.get('data', 'fr')
    movingAverageFrames = round(movingAverage * frameRate)  # TODO: be a # instead to slice cnm
    baseFrames = round(baseLength * frameRate)  # TODO: CHECK IF THIS IS FOR BASELINE
    relaxationFrames = round(relaxationTime * frameRate)
    timeoutFrames = round(timeout * frameRate)

    # %% ONSET INITIALIZATION

    # expHistory = np.empty((units, movingAverageFrames), dtype=np.float32)  # define a windows buffer
    cursor = np.zeros(expectedLengthExperiment, dtype=np.float32)  # define a very long vector for cursor
    frequency = np.zeros(expectedLengthExperiment, dtype=np.float32)  # TODO: CHECK IF OK TO REMOVE
    baseval = np.ones(units, dtype=np.float32)
    i = 0 # TODO: CHECK AGAIN AND SET TO 0
    rewardHistory = 0
    trialHistory = 0
    #motionFlag = False
    motionCounter = 0
    lastVol = 0  # careful with this it may create problems
    tim = time.time()  # TODO: CHECK USAGE
    trialFlag = True
    counter = 40
    backtobaselineFlag = False
    misscount = 0  # TODO: CHECK IF OK MOVING EVERYTHING IN THE FRONT

    # %% saving fields to cnm.param
    cnm.params.bmi = {
        'cursor': cursor,
        'frequency': frequency,
        'miss': [],
        'hits': [],
        'trialEnd': [],
        'trialStart': [],
        'duration': i,
        'curval': 0,
        'freqval': 0,
        'dynamicLen': movingAverageFrames,
        'E1': E1,
        'E2': E2
    }
    if not sim:
        a = ArduinoDelegate(port=SETTINGS.PORT)
    print('starting arduino')  # TODO: CHECK TO HANDLE CONNECTION LOSS

    def feed_to_bmi(allvals, *args):
        # TODO: FFT/LOWPASS IMPLEMENT
        #nonlocal a TODO: ARDUINO
        nonlocal rewardHistory
        nonlocal trialHistory
        nonlocal i
        nonlocal tim
        #nonlocal motionFlag
        nonlocal motionCounter
        nonlocal trialFlag
        nonlocal backtobaselineFlag
        nonlocal lastVol
        nonlocal baseval
        nonlocal counter
        nonlocal misscount
        nonlocal T1
        # nonlocal expHistory TODO: WE MIGHT NOT NEED THIS
        nonlocal cursor  # [0, 1] E1, [2, 3] E2
        nonlocal frequency

        # %%**************************************************************
        # *************************** RUN ********************************
        # ****************************************************************
        # TODO: ONACID CURRENTLY DOES NOT REQUIRE PLACEHOLDER
        # TODO: FIND OUT IF REWARD DELIVERED DURING BASELINE? BASEVAL?
        # TODO: CALCULATE ONLY AFTER CERTAIN PERIOD
        # THIS chunk of code simply updates the expHistory which is a buffer of frames on which
        # we calculate average, in our case we only need a length, and we could slices on

        vals = allvals[rois]

        i += 1  # TODO: IF SUPPORT DISCARD VALUES WHEN TIMEOUT, STEPS INSTEAD, RESET CHECKING

        # handle misscount
        if misscount > 9:
            T1 = T1 - T1 / 20
            print(['New T1: ', str(T1)])
            misscount = 0

        # handle motion
        # because we don't want to stim or reward if there is motion
        """
        mot = evalin('base', 'hSI.hMotionManager.motionCorrectionVector') # TODO: CHECK USAGE
        if mot:
            motion = np.sqrt(mot(1)^2 + mot(2)^2 + mot(3)^2)
        else:
            motion = 0


        if motion>motionThresh:
            motionFlag = 1
            motionCounter = 0

        if motionFlag == 1:     # flag if there was motion "motionRelaxationFrames" ago, do not allow reward
           motionCounter = motionCounter + 1
           if motionCounter>=motionRelaxationFrames:
               motionFlag = 0"""  # TODO: HANDLING PAST MOTION

        if counter == 0:
            # Is it a new trial?
            if trialFlag and not backtobaselineFlag:
                cnm.params.get('bmi', 'trialStart').append(i)  # TODO: Check
                # usage
                trialHistory += 1
                trialFlag = False
                # start running the timer again
                tim = time.time()  # TODO: CHECK USAGE
                print('New Trial!')
            # calculate baseline activity and actual activity for the DFF
            # signal = np.nanmean(expHistory, 1) # TODO: Check Usage √
            signal = np.nanmean(vals, 1)
            if debug:
                cnm.raw_sig[:, i-1] = signal
            if std_filter_thres:
                std_ftr.add(signal)
                baseval = std_ftr.get_val()
            else:
                if i >= baseFrames:
                    baseval = (baseval * (baseFrames - 1) + signal) / baseFrames  # TODO: CHECK USAGE TO SEE IF DOT PRODUCT
                else:
                    baseval = (baseval * (i - 1) + signal) / i
            if debug:
                cnm.base_vals[:, i-1] = baseval

            print(i)

            # calculation of DFF
            dff = (signal - baseval) / baseval  # TODO: VERIFY WHY CALCULATION IS AFTER NEW SIGNAL TAKEN INTO BASEVAL
            dff[dff < T1 * ens_thresh] = T1 * ens_thresh  # limit the contribution of each neuron to a portion of the target
            # it is very unprobable (almost imposible) that one cell of E2 does
            # it on its own, buuut just in case:
            dff[dff > -T1 * ens_thresh] = -T1 * ens_thresh  # T1 is negative and thereby the effect

            cursor[i-1] = np.nansum([np.nansum(dff[E1]), -np.nansum(dff[E2])])  # TODO: WHY USING NANSUM?

            # calculate frequency
            freq = np.around(freqmed + 2000 - cursor[i-1] / T1 * freqmed).astype(
                np.float64)  # FREQUENCY RISE WHEN HIT REWARD
            if np.isnan(freq) or freq < 0:  # TODO: SHOULDN'T NAN BE +INF?, which is freqMax
                freq = 0
            elif freq > freqMax:  # this shouldnt happen because it would be a hit
                freq = freqMax

            frequency[i-1] = freq

            if backtobaselineFlag:
                if cursor[i-1] >= 1 / 2 * T1:
                    backtobaselineFlag = False
                tim = time.time()  # to avoid false timeouts while it goes back to baseline
            else:
                #if cursor[i-1] <= T1 and not motionFlag:  # if it hit the target
                if cursor[i - 1] <= T1:
                    # remove tone
                    if not sim:
                        a.playTone(freq)
                        #print('Tone played {}'.format(freq))
                        # give water reward
                        a.reward() #TODO: SWITCH TO 1 FOR TTL
                    time.sleep(0.01)
                    # update rewardHistory
                    rewardHistory = rewardHistory + 1
                    print(['Trial: ', str(trialHistory), 'Rewards: ', str(rewardHistory)])
                    # update trials and hits vector
                    cnm.params.get('bmi', 'trialEnd').append(i)
                    cnm.params.get('bmi', 'hits').append(i)
                    trialFlag = True
                    misscount = 0
                    counter = relaxationFrames
                    backtobaselineFlag = True
                else:
                    # update the tone to the new cursor
                    if not sim:
                        a.playTone(freq)
                    #print('Tone played {}'.format(freq))
                    if time.time() - tim > durationTrial:
                        print('Timeout')
                        cnm.params.get('bmi', 'trialEnd').append(i)
                        cnm.params.get('bmi', 'miss').append(i)
                        trialFlag = True
                        misscount += 1
                        counter = timeoutFrames
                    """if cursor[i-1] >= T1 and motionFlag:
                        print('mot too high for reward')"""
        else:
            counter = counter - 1

        ## Outputs
        curval = cursor[i-1]
        freqval = frequency[i-1]
        cnm.params.change_params({'curval': curval,
                                  'freqval': freqval,
                                  'duration': i})
        cnm.params.get('bmi', 'cursor')[i-1] = curval
        cnm.params.get('bmi', 'frequency')[i - 1] = freqval
        if not sim and i == expectedLengthExperiment:
            a.close()

    cnm.feed_to_bmi = feed_to_bmi


def baselineSimulation(cnm, E1=None, E2=None, *args, **kwargs):
    # Function to obtain the value of T1 when there is no end of the trial
    # kwargs: lPC: tuple [float, float], LowerBound & UpperBound of
    # Percentage Correct
    if E1 is None:
        E1 = [2, 3]
    if E2 is None:
        E2 = [0, 1]

    # initialize variables
    T1 = -10
    maxiter = 20
    percentCorrect = 0
    if 'lPC' in kwargs.keys():
        limitPercentCorrect = kwargs['lPC']
    else:
        limitPercentCorrect = [25, 40] # 33 + - 5
    trialDuration = 30
    frameRate = cnm.params.get('data', 'fr')
    relaxationTime = 5 # sec
    perT1 = 0.3 # initial percentil to obtain T1
    relaxationFrames = round(relaxationTime * frameRate) # hSI.hRoiManager.scanFrameRate)
    baselineVector = cnm.baseline


    numberTrials = round(cnm.base / frameRate / trialDuration)
    # first inital cursor with T1 by default to initialize the cursor
    cursor = obtainCursor(cnm, E1, E2, T1, frameRate)
    iter = 0
    tol = 0.1

    while (percentCorrect > limitPercentCorrect[1]) or (percentCorrect < limitPercentCorrect[0]):
        iter += 1
        if iter >= 10:
            tol = 0.5
            if iter >= maxiter:
                print('Reaching Max Iterations ...')
                print(percentCorrect)
                return None, None


        T1 = np.percentile(cursor, perT1) # in (#)
        cursor = obtainCursor(cnm, E1, E2, T1, frameRate)
        actBuffer = np.zeros((relaxationFrames, 1))

        success = 0

        #print(cursor)

        for ind in range(len(cursor)):
            activationAUX = 0
            if cursor[ind] <= T1 and sum(actBuffer) == 0:
                activationAUX = 1
                success = success + 1


            actBuffer[:-1] = actBuffer[1:]
            actBuffer[-1] = activationAUX


        percentCorrect = success / numberTrials * 100
        if percentCorrect > limitPercentCorrect[1]:
            perT1 = perT1-tol
        if percentCorrect < limitPercentCorrect[0]:
            perT1 = perT1+tol
        print("Percentage Correct {}, T1: {}".format(str(percentCorrect), str(T1)))
        if perT1 >= 100 or perT1 <= 0:
            print("Warning imposible to initiate T1")
            return None, None

    # figure()
    # plot(cursor / T1)
    # xlabel('cursor')
    # figure()
    # freqmed = (18000 - 2000) / 2
    # freqplot = double(round(freqmed + 2000 + cursor / T1 * freqmed))
    # plot(freqplot)
    # xlabel('freq')
    print(T1)
    return percentCorrect, T1


def obtainCursor(cnm, E1=None, E2=None, T1=None, *args):

    if T1 is None:
        T1 = -100  # TODO: CHECK USAGE
    if E1 is None:
        E1 = [0, 1]
    if E2 is None:
        E2 = [2, 3]

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # these are parameters that we will play with and eventually not change
    baseLength = 2 * 60 # seconds for baseline period

    baselineVector = cnm.baseline

    frameRate = cnm.params.get('data', 'fr')

    movingAverage = int(np.ceil(1 * frameRate)) # TODO: we may want to make these guys global

    ens_thresh = 0.95

    baseFrames = round(baseLength * frameRate)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    cursor = np.zeros((cnm.base, 1))
    baseval = 0
    # First pass at T1=1000 so every bit counts
    for ind in range(cnm.base):
        if ind >= movingAverage: # TODO: CHECK USAGE WHETHER ind should be strictly greater than movingAv
            signal = np.nanmean(baselineVector[:, ind - movingAverage: ind+1], 1)
        else:
            signal = np.nanmean(baselineVector[:, :ind+1], 1)

        if ind >= baseFrames:
            baseval = (baseval * (baseFrames - 1) + signal) / baseFrames
        else:
            baseval = (baseval * ind + signal) / (ind + 1)

        #print('sig', signal, baseval)
        dff = (signal - baseval) / baseval
        dff[dff < T1 * ens_thresh] = T1 * ens_thresh # limit the contribution of each neuron to a portion of the target
        # it is very unprobable (almost imposible) that one cell of E2 does
        # it on its own, buuut just in case:
        dff[dff > -T1 * ens_thresh] = -T1 * ens_thresh

        cursor[ind] = np.nansum([np.nansum(dff[E1]), -np.nansum(dff[E2])])


    cursor = cursor[cursor != 0] # TODO: CHECK USAGE
    # we want 1/3 of the times in 30 sec trials so once in 90sec ~ 1 in 100sec
    # with the framerate being ~10
    return cursor


