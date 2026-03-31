import re
from prompt_pool import ANSWER_PREFIX


class AnswerParsing:
    def __init__(self, dataset):
        self.dataset = dataset

    def dataset_parse(self, pred, true, sample):
        if "gsm" in self.dataset:
            extracted_answer, binary = self.mgsm_parse(ANSWER_PREFIX["en"], pred, true)
        elif "commonsenseqa" in self.dataset:
            extracted_answer, binary = self.commonsenseqa_parse(pred, true)
        elif "mmmlu" in self.dataset:
            extracted_answer, binary = self.mmmlu_parse(pred, true)
        elif "belebele" in self.dataset:
            extracted_answer, binary = self.belebele_parse(pred, true)
        elif self.dataset == "answer":
            extracted_answer, binary = self.answer_parse(pred, true)

        return extracted_answer, binary




    def gsm8k_parse(self, pred, true):
        if "<|im_end|>" not in pred and "<|eot_id|>" not in pred and "</s>" not in pred and "<|END_OF_TURN_TOKEN|>" not in pred:
            return "Incomplete", False

        extracted_answer = self.extract_numbers_from_boxed(pred)

        extracted_answer = extracted_answer[0] if extracted_answer else None
        if extracted_answer:
            extracted_answer = extracted_answer.replace(" ", "")
            true = true.replace(" ", "")
        # if true == extracted_answer:
        #     print(pred[0:100])

        return extracted_answer, true == extracted_answer

    def extract_numbers_from_boxed(self, text):
        # matches = re.findall(r'\\boxed\{([^}]*)\}', text)
        matches = re.findall(r'\\?boxed\{([^}]*)\}', text)

        cleaned_matches = []
        for m in matches:
            # 去掉 $ , 空格等，只保留数字和小数点
            cleaned = re.sub(r'[^\d.]', '', m)
            cleaned_matches.append(cleaned)
        return cleaned_matches





    def answer_parse(self, pred, true):
        if "<|im_end|>" not in pred and "<|eot_id|>" not in pred and "</s>" not in pred and "<|END_OF_TURN_TOKEN|>" not in pred:
            return "Incomplete", False

        extracted_answer = self.extract_answer_content(pred)
        extracted_answer = extracted_answer[0] if extracted_answer else None
        if extracted_answer:
            extracted_answer = extracted_answer.replace(" ", "")
            true = true.replace(" ", "")

        return extracted_answer, true == extracted_answer

    def extract_answer_content(self, text):
        pattern = re.compile(r'\\answer{')
        matches = pattern.finditer(text)
        results = []
        for match in matches:
            start_pos = match.end()
            brace_count = 1
            i = start_pos
            while i < len(text) and brace_count > 0:
                if text[i] == '{':
                    brace_count += 1
                elif text[i] == '}':
                    brace_count -= 1
                i += 1
            if brace_count == 0:
                results.append(text[start_pos:i-1])
        return results



    def mmmlu_parse(self, pred, true):
        if "<|im_end|>" not in pred and "<|eot_id|>" not in pred and "</s>" not in pred and "<|END_OF_TURN_TOKEN|>" not in pred:
            return "Incomplete", False

        ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer\s*:\s*([A-D])"
        pred = pred.replace("$", "")
        pred = pred.replace("(", "")
        pred = pred.replace(")", "")
        match = re.search(ANSWER_PATTERN_MULTICHOICE, pred)
        extracted_answer = match.group(1) if match else None
            
        return extracted_answer, true == extracted_answer


    def commonsenseqa_parse(self, pred, true):
        if "<|im_end|>" not in pred and "<|eot_id|>" not in pred and "</s>" not in pred and "<|END_OF_TURN_TOKEN|>" not in pred:
            return "Incomplete", False

        ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer\s*:\s*([A-E])"
        pred = pred.replace("$", "")
        pred = pred.replace("(", "")
        pred = pred.replace(")", "")
        match = re.search(ANSWER_PATTERN_MULTICHOICE, pred)
        extracted_answer = match.group(1) if match else None
        
        return extracted_answer, true == extracted_answer


    def belebele_parse(self, pred, true):
        alpha_map = ["", "A", "B", "C", "D"]
        if "<|im_end|>" not in pred and "<|eot_id|>" not in pred and "</s>" not in pred and "<|END_OF_TURN_TOKEN|>" not in pred:
            return "Incomplete", False

        ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer\s*:\s*([A-D])"
        pred = pred.replace("$", "")
        pred = pred.replace("(", "")
        pred = pred.replace(")", "")
        match = re.search(ANSWER_PATTERN_MULTICHOICE, pred)
        extracted_answer = match.group(1) if match else None

        return extracted_answer, true == extracted_answer
