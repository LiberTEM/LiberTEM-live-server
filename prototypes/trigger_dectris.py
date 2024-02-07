from libertem_live.detectors.dectris.DEigerClient import DEigerClient


if __name__ == "__main__":
    ec = DEigerClient("localhost", port=8910)
    result = ec.sendDetectorCommand('arm')
    result = ec.sendDetectorCommand('trigger')
