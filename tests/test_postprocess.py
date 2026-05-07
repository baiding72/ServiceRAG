from app.postprocess import format_competition_answer, normalize_image_alignment, postprocess_answer


def test_image_alignment_removes_illegal_images():
    text, images = normalize_image_alignment("状态A <PIC> 状态B <PIC>", ["img1", "bad"], allowed_images=["img1"])
    assert images == ["img1"]
    assert text.count("<PIC>") == 1


def test_no_images_removes_pic():
    text, images = postprocess_answer("请看 <PIC>", [], allowed_images=[])
    assert "<PIC>" not in text
    assert images == []


def test_competition_answer_format():
    assert format_competition_answer("回答 <PIC>", ["a"]) == '回答 <PIC> , ["a"]'

