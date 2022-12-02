"""
Define default way of dealing with columns for the data.
_FEATURES:  are ready to be used for clustering/prediction out the box
_FEATURES_TO_BE_TIME_ENCODED: columns need to be converted into unix time format (intervals should be per quarter hour)
_FEATURES_TO_BE_OR_ENCODED: columns need to be ordinally encoded
_FEATURES_TO_BE_OH_ENCODED: columns need to be one-hot encoded
_FEATURES_META: contains additional information on the object not directly relevant to the clustering/prediction tasks 
"""


class U_features():
    def __init__(self,
                 features=['user_grade', 'has_teacher_cnt', 'has_student_cnt', 'has_class_cnt',
                           "correct_percentage", "problems_attempted", "average_level", "max_level",
                           "average_hints", "avg_difficulty", "avg_learning_stage"],
                 features_to_be_time_encoded=['first_login_date_TW'], features_to_be_OR_encoded=[],
                 features_to_be_OH_encoded=['gender', 'user_city', 'is_self_coach'], features_meta=['uuid'],
                 features_to_be_scaled=['points', "correct_percentage", "time_spent", 'belongs_to_class_cnt',
                                        "badges_cnt"]):
        self.features = features
        self.features_to_be_time_encoded = features_to_be_time_encoded
        self.features_to_be_scaled = features_to_be_scaled
        self.features_to_be_OR_encoded = features_to_be_OR_encoded
        self.features_to_be_OH_encoded = features_to_be_OH_encoded
        self.features_meta = features_meta


class Pr_features():
    def __init__(self,
                 features=['problem_number', 'exercise_problem_repeat_session', 'total_sec_taken', 'total_attempt_cnt',
                           'used_hint_cnt', 'level'], features_to_be_time_encoded=['timestamp_TW'],
                 features_to_be_OR_encoded=[],
                 features_to_be_OH_encoded=['is_correct', 'is_hint_used', 'is_downgrade', 'is_upgrade'],
                 features_meta=['uuid', 'ucid', 'upid']):
        self.features = features
        self.features_to_be_scaled = ['total_sec_taken']
        self.features_to_be_time_encoded = features_to_be_time_encoded
        self.features_to_be_OR_encoded = features_to_be_OR_encoded
        self.features_to_be_OH_encoded = features_to_be_OH_encoded
        self.features_meta = features_meta


class Ex_features():
    def __init__(self, features=[], features_to_be_time_encoded=[],
                 features_to_be_OR_encoded=['difficulty', 'learning_stage'],
                 features_to_be_OH_encoded=['level2_id', 'level3_id', 'level4_id'], features_meta=['ucid']):
        self.features = features
        self.features_to_be_time_encoded = features_to_be_time_encoded
        self.features_to_be_scaled = []
        self.features_to_be_OR_encoded = features_to_be_OR_encoded
        self.features_to_be_OH_encoded = features_to_be_OH_encoded
        self.features_meta = features_meta


if __name__ == "__main__":
    # df_u, df_pr, df_ex = load_data_raw(subset=True)
    pass
