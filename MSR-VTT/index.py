from sys import path
path.append("../")

import search
import constants


if __name__ == "__main__":
    search.index(constants.MSR_VTT_TEST_VIDEO_DIR)
