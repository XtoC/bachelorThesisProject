import scaper
import numpy as np
import os
import csv
import jams

Alarm_types = ["Smoke_alarm", "Fire_alarm_bell", "Fire_alarm_electronic", "Air_siren"]

def sound_mix (outfolder, fg_folder, bg_folder, fg_type, n_soundscapes, snr_min, snr_max, start_id, filename):
    logfile = os.path.join(outfolder, "soundscape_log.csv")

    # Write header once
    with open(logfile, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "soundscapes", "bg_label", "bg_filesource", "fg_label", "fg_filesource", "snr"])

    # OUTPUT FOLDER
    # outfolder = 'audio/soundscapes/Smoke_Alarm'

    # SCAPER SETTINGS
    # fg_folder = 'audio/foreground_audio/'
    # bg_folder = 'audio/background_audio/'
    file_folder = os.path.join(os.path.join(fg_folder, fg_type), filename)

    # n_soundscapes = 1
    ref_db = -50
    duration = 10.0

    min_events = 1
    max_events = 9

    event_time_dist = 'truncnorm'
    event_time_mean = 5.0
    event_time_std = 2.0
    event_time_min = 0.0
    event_time_max = 10.0

    source_time_dist = 'const'
    source_time = 0.0

    event_duration_dist = 'uniform'
    event_duration_min = 0.5
    event_duration_max = 4.0

    snr_dist = 'uniform'
    # snr_min = 20
    # snr_max = 30

    pitch_dist = 'uniform'
    pitch_min = -3.0
    pitch_max = 3.0

    time_stretch_dist = 'uniform'
    time_stretch_min = 0.8
    time_stretch_max = 1.2

    # Generate 1000 soundscapes using a truncated normal distribution of start times

    for n in range(n_soundscapes):

        print('Generating soundscape: {:d}/{:d}'.format(n + 1, n_soundscapes))

        # create a scaper
        sc = scaper.Scaper(duration, fg_folder, bg_folder)
        sc.protected_labels = []
        sc.ref_db = ref_db

        # add background
        sc.add_background(label=('choose', []),
                          source_file=('choose', []),
                          source_time=('const', 0))

        # add random number of foreground events
        n_events = 1
        for _ in range(n_events):
            sc.add_event(label=('const', fg_type),
                         source_file=('const', file_folder),
                         source_time=(source_time_dist, source_time),
                         event_time=(event_time_dist, event_time_mean, event_time_std, event_time_min, event_time_max),
                         event_duration=(event_duration_dist, event_duration_min, event_duration_max),
                         snr=(snr_dist, snr_min, snr_max),
                         pitch_shift=(pitch_dist, pitch_min, pitch_max),
                         time_stretch=(time_stretch_dist, time_stretch_min, time_stretch_max))

        # generate
        audiofile = os.path.join(outfolder, "soundscape_unimodal{:d}.wav".format(start_id+n))
        jamsfile = os.path.join(outfolder, "soundscape_unimodal{:d}.jams".format(start_id+n))
        txtfile = os.path.join(outfolder, "soundscape_unimodal{:d}.txt".format(start_id+n))

        sc.generate(audiofile, jamsfile,
                    allow_repeated_label=True,
                    allow_repeated_source=False,
                    reverb=0.1,
                    disable_sox_warnings=True,
                    no_audio=False,
                    txt_path=txtfile)

        # ---- Extract SNRs (and more) from the JAMS we just wrote ----
        jam = jams.load(jamsfile)
        ann = jam.annotations.search(namespace='scaper')[0]

        fg_filesource = None
        bg_filesource = None
        bg_label = None
        fg_label = None
        snr_val = None

        for obs in ann.data:
            val = obs.value
            role = val.get('role')

            if role == 'background' and bg_filesource is None:
                bg_label = val.get('label')
                bg_filesource = val.get('source_file')

            if role == 'foreground':
                fg_label = val.get('label')
                fg_filesource = val.get('source_file')
                snr_val = val.get('snr')


        # Append to CSV
        with open(logfile, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([start_id+n, "soundscape_unimodal{:d}.wav".format(start_id+n), bg_label, bg_filesource, fg_label, fg_filesource, snr_val])

def generate_train_data():
    for alarm_type in Alarm_types:
        outfolder = os.path.join("audio/soundscapes/train_audio/", alarm_type)
        sound_mix(outfolder, "audio/foreground_audio/train_audio", "audio/brownian_noise", alarm_type,
                  20, 20, 30, 1)
        sound_mix(outfolder, "audio/foreground_audio/train_audio", "audio/brownian_noise", alarm_type,
                  20, 6, 10, 21)

        sound_mix(outfolder, "audio/foreground_audio/train_audio", "audio/background_audio/train_audio", alarm_type,
                  50, 20, 30, 41)
        sound_mix(outfolder, "audio/foreground_audio/train_audio", "audio/background_audio/train_audio", alarm_type,
                  50, 6, 10, 91)

def generate_test_data():
    test_folder = "audio/foreground_audio/test_audio"
    for dir in os.listdir(test_folder):
        sub_dir = os.path.join(test_folder, dir)
        # print(dir, sub_dir)
        outfolder = os.path.join("audio/soundscapes/test_audio", dir)
        if os.path.isdir(sub_dir):
            start_id = 1
            for filename in os.listdir(sub_dir):
                if filename == ".DS_Store":
                    continue
                sound_mix(outfolder, "audio/foreground_audio/test_audio", "audio/brownian_noise", dir,
                          1, 20, 30, start_id, filename)
                start_id += 1

                sound_mix(outfolder, "audio/foreground_audio/test_audio", "audio/brownian_noise", dir,
                          1, 6, 10, start_id, filename)
                start_id += 1

                sound_mix(outfolder, "audio/foreground_audio/test_audio", "audio/background_audio/test_audio", dir,
                          2, 20, 30, start_id, filename)
                start_id += 2

                sound_mix(outfolder, "audio/foreground_audio/test_audio", "audio/background_audio/test_audio", dir,
                          2, 6, 10, start_id, filename)
                start_id += 2

def generate_validation_data():
    validation_folder = "audio/foreground_audio/validation_audio"
    for dir in os.listdir(validation_folder):
        sub_dir = os.path.join(validation_folder, dir)
        # print(dir, sub_dir)
        outfolder = os.path.join("audio/soundscapes/validation_audio", dir)
        if os.path.isdir(sub_dir):
            start_id = 1
            for filename in os.listdir(sub_dir):
                if filename == ".DS_Store":
                    continue
                sound_mix(outfolder, "audio/foreground_audio/validation_audio", "audio/brownian_noise", dir,
                          1, 20, 30, start_id, filename)
                start_id += 1

                sound_mix(outfolder, "audio/foreground_audio/validation_audio", "audio/brownian_noise", dir,
                          1, 6, 10, start_id, filename)
                start_id += 1

                sound_mix(outfolder, "audio/foreground_audio/validation_audio", "audio/background_audio/validation_audio", dir,
                          2, 20, 30, start_id, filename)
                start_id += 2

                sound_mix(outfolder, "audio/foreground_audio/validation_audio", "audio/background_audio/validation_audio", dir,
                          2, 6, 10, start_id, filename)
                start_id += 2

if __name__ == "__main__":
    # sound_mix("audio/soundscapes/train_audio/Smoke_alarm", "audio/foreground_audio/train_audio",
    #           "audio/brownian_noise", "Smoke_alarm", 20, 20, 30, 1)
    # sound_mix("audio/soundscapes/train_audio/Smoke_alarm", "audio/foreground_audio/train_audio",
    #           "audio/brownian_noise", "Smoke_alarm", 20, 6, 10, 21)
    #
    # sound_mix("audio/soundscapes/train_audio/Smoke_alarm", "audio/foreground_audio/train_audio",
    #           "audio/background_audio/train_audio", "Smoke_alarm", 50, 20, 30, 41)
    # sound_mix("audio/soundscapes/train_audio/Smoke_alarm", "audio/foreground_audio/train_audio",
    #           "audio/background_audio/train_audio", "Smoke_alarm", 50, 6, 10, 91)

    #generate_train_data()
    #generate_test_data()
    generate_validation_data()