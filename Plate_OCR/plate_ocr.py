class PlateReader:
    def __init__(self, backend: OCRBackend, height_width_ratio_floor: float = 0.55,
                 deblur_backend: DeblurBackend | None = None,
                 retry_confidence_threshold: float = 0.5, blur_score_threshold: float = 60.0):
        self.backend = backend
        self.height_width_ratio_floor = height_width_ratio_floor
        self.deblur_backend = deblur_backend
        self.retry_confidence_threshold = retry_confidence_threshold
        self.blur_score_threshold = blur_score_threshold

    def _read_once(self, plate_crop: np.ndarray) -> tuple[str, float, bool]:
        if _looks_multiline(plate_crop, self.height_width_ratio_floor):
            top, bottom = _split_two_line(plate_crop)
            top_text, top_conf = self.backend.read_text(top)
            bottom_text, bottom_conf = self.backend.read_text(bottom)
            texts = [t for t in (top_text, bottom_text) if t]
            confs = [c for t, c in ((top_text, top_conf), (bottom_text, bottom_conf)) if t]
            combined_text = "".join(texts)
            mean_conf = float(sum(confs) / len(confs)) if confs else 0.0
            return combined_text, mean_conf, True

        text, conf = self.backend.read_text(plate_crop)
        return text, conf, False

    def read(self, plate_crop: np.ndarray) -> tuple[str, float, bool]:
        if plate_crop is None or plate_crop.size == 0:
            return "", 0.0, False

        result = self._read_once(plate_crop)

        if self.deblur_backend is None:
            return result

        _, confidence, _ = result
        if confidence >= self.retry_confidence_threshold:
            return result

        from .blur_detector import is_blurry
        if not is_blurry(plate_crop, threshold=self.blur_score_threshold):
            return result

        deblurred_crop = self.deblur_backend.deblur(plate_crop)
        retry_result = self._read_once(deblurred_crop)

        return retry_result if retry_result[1] > result[1] else result