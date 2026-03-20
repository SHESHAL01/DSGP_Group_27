import unittest
import pandas as pd
import numpy as np
from sentence_transformers import InputExample
from C4 import parse_skills, create_positive_pairs, retrieve_top_k_courses, evaluate_model_with_metrics

# ------------------- Dummy Model for Testing -------------------
class DummyModel:
    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
        # Return a simple fixed embedding based on length of text
        return np.array([[len(t), len(t)*0.5] for t in texts])

# ------------------- Test Cases -------------------
class TestCourseModel(unittest.TestCase):

    # ------------------- parse_skills -------------------
    def test_parse_skills_list(self):
        s1 = "['Python', 'Data Analysis']"
        self.assertEqual(parse_skills(s1), ['python', 'data analysis'])

        s2 = "Python, Data Analysis"
        self.assertEqual(parse_skills(s2), ['python', 'data analysis'])

        s3 = ""
        self.assertEqual(parse_skills(s3), [])

        s4 = None
        self.assertEqual(parse_skills(s4), [])

    # ------------------- create_positive_pairs -------------------
    def test_create_positive_pairs(self):
        df = pd.DataFrame({
            "Title": ["Course1", "Course2", "Course3"],
            "skills_list": [["python", "ml"], ["ml", "ai"], ["python"]]
        })

        # ADD THIS LINE
        df["combined_text"] = df.apply(
            lambda r: f"{r['Title']}. Skills: {' '.join(r['skills_list'])}", axis=1
        )

        pairs = create_positive_pairs(df, min_overlap=1)

        self.assertTrue(len(pairs) > 0)

        for p in pairs:
            self.assertIsInstance(p, InputExample)
            self.assertTrue(hasattr(p, "texts"))
            self.assertEqual(len(p.texts), 2)

    # ------------------- retrieve_top_k_courses -------------------
    def test_retrieve_top_k_courses(self):
        df = pd.DataFrame({
            "Title": ["Course1", "Course2", "Course3"],
            "Url": ["url1", "url2", "url3"]
        })
        embeddings = np.array([[1, 0], [0, 1], [1, 1]])
        model = DummyModel()
        top_courses = retrieve_top_k_courses(model, df, embeddings, "python", k=2)
        self.assertEqual(len(top_courses), 2)
        self.assertTrue("Title" in top_courses.columns)
        self.assertTrue("Url" in top_courses.columns)

    # ------------------- evaluate_model_with_metrics -------------------
    # def test_evaluate_model_with_metrics(self):
    #     df = pd.DataFrame({
    #         "Title": ["C1", "C2"],
    #         "Skills": ["['python']", "['python', 'ml']"]
    #     })
    #     df["skills_list"] = df["Skills"].apply(parse_skills)
    #     df["combined_text"] = df.apply(lambda r: f"{r['Title']}. Skills: {' '.join(r['skills_list'])}", axis=1)
    #
    #     model = DummyModel()
    #     metrics = evaluate_model_with_metrics(model, df, sample_size=4)
    #     # Check that all metric keys exist
    #     self.assertTrue(all(k in metrics for k in ["AP", "ROC-AUC", "F1", "Precision", "Recall"]))

# ------------------- Run Tests -------------------
if __name__ == "__main__":
    unittest.main()