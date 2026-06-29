import re
from pathlib import Path

def main():
    train_path = Path("src/football_ai/train.py")
    content = train_path.read_text(encoding="utf-8")
    
    # 1. Extract save_patched and move to global scope, with correct indentation
    match = re.search(r"    def save_patched\(artifact_obj, dest_path\):", content)
    if not match:
        print("Error: Could not find save_patched definition.")
        return
    start_idx = match.start()

    end_pattern = r"        with open\(dest_path, \"wb\"\) as f:\n            f\.write\(bytecode\)"
    end_match = re.search(end_pattern, content[start_idx:])
    if not end_match:
        print("Error: Could not find the end of save_patched.")
        return
    end_idx = start_idx + end_match.end()

    save_patched_code = content[start_idx:end_idx]
    
    # De-indent lines, but DO NOT de-indent lines inside the f-string payload
    de_indented_lines = []
    in_payload = False
    for line in save_patched_code.splitlines():
        if 'payload_code = f"""exec(\'\'\'' in line:
            in_payload = True
            de_indented_lines.append(line[4:] if line.startswith("    ") else line)
        elif 'loaded_model"""' in line:
            in_payload = False
            de_indented_lines.append(line[4:] if line.startswith("    ") else line)
        elif in_payload:
            # Leave payload string indentation untouched!
            de_indented_lines.append(line)
        else:
            de_indented_lines.append(line[4:] if line.startswith("    ") else line)
            
    global_save_code = "\n".join(de_indented_lines)
    global_save_code = global_save_code.replace("def save_patched(artifact_obj, dest_path):", "def save_model_artifact(artifact_obj, dest_path):")

    # Replace local save_patched block in train_model
    new_content = content[:start_idx] + "    def save_patched(artifact_obj, dest_path):\n        save_model_artifact(artifact_obj, dest_path)" + content[end_idx:]

    # Add save_model_artifact at global scope before train_model
    train_model_match = re.search(r"def train_model\(", new_content)
    if not train_model_match:
        print("Error: Could not find train_model definition.")
        return
    insert_idx = train_model_match.start()
    
    final_content = new_content[:insert_idx] + global_save_code + "\n\n\n" + new_content[insert_idx:]

    # 2. Modify local PredictorWrapper.predict to use neutral venue expected goals
    target_local_predict = (
        '        home_goals_float = float(self.home_goals_model.predict(features)[0])\n'
        '        away_goals_float = float(self.away_goals_model.predict(features)[0])\n'
        '        home_goals_float = max(0.01, home_goals_float)\n'
        '        away_goals_float = max(0.01, away_goals_float)'
    )
    replacement_local_predict = (
        '        # Neutral venue expected goals averaging (FIFA World Cup)\n'
        '        frame_swapped = pd.DataFrame([{\n'
        '            "home_elo": get_val(away, "elo"),\n'
        '            "away_elo": get_val(home, "elo"),\n'
        '            "home_elo_rank": get_val(away, "elo_rank"),\n'
        '            "away_elo_rank": get_val(home, "elo_rank"),\n'
        '        }])\n'
        '        frame_swapped["elo_diff"] = frame_swapped["home_elo"] - frame_swapped["away_elo"]\n'
        '        frame_swapped["rank_diff"] = frame_swapped["away_elo_rank"] - frame_swapped["home_elo_rank"]\n'
        '        features_swapped = frame_swapped[feature_cols]\n\n'
        '        goals_a_as_home = float(self.home_goals_model.predict(features)[0])\n'
        '        goals_a_as_away = float(self.away_goals_model.predict(features_swapped)[0])\n'
        '        goals_b_as_away = float(self.away_goals_model.predict(features)[0])\n'
        '        goals_b_as_home = float(self.home_goals_model.predict(features_swapped)[0])\n\n'
        '        home_goals_float = (goals_a_as_home + goals_a_as_away) / 2.0\n'
        '        away_goals_float = (goals_b_as_away + goals_b_as_home) / 2.0\n\n'
        '        home_goals_float = max(0.01, home_goals_float)\n'
        '        away_goals_float = max(0.01, away_goals_float)'
    )
    
    # We replace only the FIRST occurrence (the local class)
    final_content = final_content.replace(target_local_predict, replacement_local_predict, 1)

    # 3. Modify pickled PredictorWrapper.predict inside save_model_artifact (the f-string payload)
    target_pickled_predict = (
        '        home_goals_float = float(self.home_goals_model.predict(features)[0])\n'
        '        away_goals_float = float(self.away_goals_model.predict(features)[0])\n'
        '        home_goals_float = max(0.01, home_goals_float)\n'
        '        away_goals_float = max(0.01, away_goals_float)'
    )
    replacement_pickled_predict = (
        '        # Neutral venue expected goals averaging (FIFA World Cup)\n'
        '        frame_swapped = pd.DataFrame([{{\n'
        '            "home_elo": get_val(away, "elo"),\n'
        '            "away_elo": get_val(home, "elo"),\n'
        '            "home_elo_rank": get_val(away, "elo_rank"),\n'
        '            "away_elo_rank": get_val(home, "elo_rank"),\n'
        '        }}])\n'
        '        frame_swapped["elo_diff"] = frame_swapped["home_elo"] - frame_swapped["away_elo"]\n'
        '        frame_swapped["rank_diff"] = frame_swapped["away_elo_rank"] - frame_swapped["home_elo_rank"]\n'
        '        features_swapped = frame_swapped[feature_cols]\n\n'
        '        goals_a_as_home = float(self.home_goals_model.predict(features)[0])\n'
        '        goals_a_as_away = float(self.away_goals_model.predict(features_swapped)[0])\n'
        '        goals_b_as_away = float(self.away_goals_model.predict(features)[0])\n'
        '        goals_b_as_home = float(self.home_goals_model.predict(features_swapped)[0])\n\n'
        '        home_goals_float = (goals_a_as_home + goals_a_as_away) / 2.0\n'
        '        away_goals_float = (goals_b_as_away + goals_b_as_home) / 2.0\n\n'
        '        home_goals_float = max(0.01, home_goals_float)\n'
        '        away_goals_float = max(0.01, away_goals_float)'
    )
    # The second occurrence in the file will be inside save_model_artifact. Let's find it.
    idx = final_content.find(target_pickled_predict)
    if idx != -1:
        final_content = final_content[:idx] + replacement_pickled_predict + final_content[idx+len(target_pickled_predict):]

    # 4. Inject grid search and update LogisticRegression/PoissonRegressor model declarations
    target_models_setup = (
        '    dev_match_model = create_pipeline(\n'
        '        LogisticRegression(C=0.1, max_iter=1000, random_state=42)\n'
        '    )\n'
        '    dev_home_goals_model = create_pipeline(PoissonRegressor(alpha=1.0))\n'
        '    dev_away_goals_model = create_pipeline(PoissonRegressor(alpha=1.0))\n'
        '    dev_btts_model = create_pipeline(\n'
        '        LogisticRegression(C=0.1, max_iter=1000, random_state=42)\n'
        '    )\n'
        '    dev_first_goal_model = create_pipeline(\n'
        '        LogisticRegression(C=0.1, max_iter=1000, random_state=42)\n'
        '    )\n'
        '    dev_home_clean_sheet_model = create_pipeline(\n'
        '        LogisticRegression(C=0.1, max_iter=1000, random_state=42)\n'
        '    )\n'
        '    dev_away_clean_sheet_model = create_pipeline(\n'
        '        LogisticRegression(C=0.1, max_iter=1000, random_state=42)\n'
        '    )'
    )
    replacement_models_setup = (
        '    # Grid search for optimal hyperparameters to maximize accuracy\n'
        '    best_c = 0.1\n'
        '    best_alpha = 1.0\n'
        '    best_score = -1.0\n'
        '    best_loss = 999.0\n\n'
        '    if not x_valid.empty:\n'
        '        for c in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]:\n'
        '            for alpha in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]:\n'
        '                test_match_model = create_pipeline(LogisticRegression(C=c, max_iter=1000, random_state=42))\n'
        '                test_home = create_pipeline(PoissonRegressor(alpha=alpha))\n'
        '                test_away = create_pipeline(PoissonRegressor(alpha=alpha))\n\n'
        '                fit_robust_classifier(test_match_model, x_train, result_train, train_weights)\n'
        '                fit_robust_regressor(test_home, x_train, y_home_goals.loc[train_index], train_weights)\n'
        '                fit_robust_regressor(test_away, x_train, y_away_goals.loc[train_index], train_weights)\n\n'
        '                try:\n'
        '                    val_home_preds = test_home.predict(x_valid)\n'
        '                    val_away_preds = test_away.predict(x_valid)\n\n'
        '                    val_probs = []\n'
        '                    val_preds = []\n'
        '                    for h_lam, a_lam in zip(val_home_preds, val_away_preds):\n'
        '                        h_pmf = stats.poisson.pmf(np.arange(15), h_lam)\n'
        '                        a_pmf = stats.poisson.pmf(np.arange(15), a_lam)\n'
        '                        h_pmf /= h_pmf.sum()\n'
        '                        a_pmf /= a_pmf.sum()\n'
        '                        joint = np.outer(h_pmf, a_pmf)\n\n'
        '                        draw_p = float(np.trace(joint))\n'
        '                        home_win_p = float(np.sum(np.tril(joint, -1)))\n'
        '                        away_win_p = float(np.sum(np.triu(joint, 1)))\n\n'
        '                        probs_dict = {"away_win": away_win_p, "draw": draw_p, "home_win": home_win_p}\n'
        '                        val_probs.append([probs_dict[cls] for cls in RESULT_LABELS])\n'
        '                        val_preds.append(max(probs_dict, key=probs_dict.get))\n\n'
        '                    val_probs = np.array(val_probs)\n'
        '                    val_accuracy = float(accuracy_score(result_valid, val_preds))\n'
        '                    val_log_loss = float(log_loss(result_valid, val_probs, labels=RESULT_LABELS))\n\n'
        '                    if (val_accuracy > best_score) or (val_accuracy == best_score and val_log_loss < best_loss):\n'
        '                        best_score = val_accuracy\n'
        '                        best_loss = val_log_loss\n'
        '                        best_c = c\n'
        '                        best_alpha = alpha\n'
        '                except Exception:\n'
        '                    pass\n\n'
        '    print(f"Optimal parameters selected: LogisticRegression C={best_c}, PoissonRegressor alpha={best_alpha} (Validation Accuracy: {best_score:.3f})")\n\n'
        '    dev_match_model = create_pipeline(\n'
        '        LogisticRegression(C=best_c, max_iter=1000, random_state=42)\n'
        '    )\n'
        '    dev_home_goals_model = create_pipeline(PoissonRegressor(alpha=best_alpha))\n'
        '    dev_away_goals_model = create_pipeline(PoissonRegressor(alpha=best_alpha))\n'
        '    dev_btts_model = create_pipeline(\n'
        '        LogisticRegression(C=best_c, max_iter=1000, random_state=42)\n'
        '    )\n'
        '    dev_first_goal_model = create_pipeline(\n'
        '        LogisticRegression(C=best_c, max_iter=1000, random_state=42)\n'
        '    )\n'
        '    dev_home_clean_sheet_model = create_pipeline(\n'
        '        LogisticRegression(C=best_c, max_iter=1000, random_state=42)\n'
        '    )\n'
        '    dev_away_clean_sheet_model = create_pipeline(\n'
        '        LogisticRegression(C=best_c, max_iter=1000, random_state=42)\n'
        '    )'
    )
    final_content = final_content.replace(target_models_setup, replacement_models_setup)

    # 5. Update prod models to use best_c and best_alpha
    target_prod_setup = (
        '    prod_match_model = create_pipeline(\n'
        '        LogisticRegression(C=0.1, max_iter=1000, random_state=42)\n'
        '    )\n'
        '    prod_home_goals_model = create_pipeline(PoissonRegressor(alpha=1.0))\n'
        '    prod_away_goals_model = create_pipeline(PoissonRegressor(alpha=1.0))\n'
        '    prod_btts_model = create_pipeline(\n'
        '        LogisticRegression(C=0.1, max_iter=1000, random_state=42)\n'
        '    )\n'
        '    prod_first_goal_model = create_pipeline(\n'
        '        LogisticRegression(C=0.1, max_iter=1000, random_state=42)\n'
        '    )\n'
        '    prod_home_clean_sheet_model = create_pipeline(\n'
        '        LogisticRegression(C=0.1, max_iter=1000, random_state=42)\n'
        '    )\n'
        '    prod_away_clean_sheet_model = create_pipeline(\n'
        '        LogisticRegression(C=0.1, max_iter=1000, random_state=42)\n'
        '    )'
    )
    replacement_prod_setup = (
        '    prod_match_model = create_pipeline(\n'
        '        LogisticRegression(C=best_c, max_iter=1000, random_state=42)\n'
        '    )\n'
        '    prod_home_goals_model = create_pipeline(PoissonRegressor(alpha=best_alpha))\n'
        '    prod_away_goals_model = create_pipeline(PoissonRegressor(alpha=best_alpha))\n'
        '    prod_btts_model = create_pipeline(\n'
        '        LogisticRegression(C=best_c, max_iter=1000, random_state=42)\n'
        '    )\n'
        '    prod_first_goal_model = create_pipeline(\n'
        '        LogisticRegression(C=best_c, max_iter=1000, random_state=42)\n'
        '    )\n'
        '    prod_home_clean_sheet_model = create_pipeline(\n'
        '        LogisticRegression(C=best_c, max_iter=1000, random_state=42)\n'
        '    )\n'
        '    prod_away_clean_sheet_model = create_pipeline(\n'
        '        LogisticRegression(C=best_c, max_iter=1000, random_state=42)\n'
        '    )'
    )
    final_content = final_content.replace(target_prod_setup, replacement_prod_setup)

    train_path.write_text(final_content, encoding="utf-8")
    print("Successfully refactored train.py with all requested updates!")

if __name__ == "__main__":
    main()
