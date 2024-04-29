from libertem_live.detectors.merlin.control import MerlinControl


if __name__ == "__main__":
    mc = MerlinControl()#"localhost", port=8910)
    with mc:
        mc.cmd("startacquisition")
        mc.cmd("softtrigger")
