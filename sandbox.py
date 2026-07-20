"""練習沙盒：專供「破壞分支 → 還原到某個 commit」練習使用。

這個檔案和主程式（model/data/train）完全無關，可以放心亂改、亂刪。
請依照《教學講義》Part C 的步驟，對這個檔案連續做幾個「破壞性」commit，
再練習用 git 回到乾淨的版本。

目前這是「乾淨基準 v1」。
"""

def broken():
    return 123


def hello():
    return "這是 conflict-demo 分支修改過的版本 v2"


if __name__ == "__main__":
    print(hello())
    print(broken())
