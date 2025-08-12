# Task Completion Checklist

## After Implementing New Features
1. **Test the implementation**:
   - Run the code to ensure it works
   - Check for any errors or warnings
   - Verify output matches expectations

2. **Code Quality Checks**:
   - Run linting if available
   - Check type hints are correct
   - Ensure PEP8 compliance

3. **Documentation**:
   - Update docstrings
   - Add comments for complex logic
   - Update README if needed

4. **Save Results**:
   - Models saved with timestamp
   - Metrics logged to results/
   - Feature importance saved
   - Plots generated and saved

5. **Version Control**:
   - Review changes with `git status`
   - Stage relevant files with `git add`
   - Commit with descriptive message
   - DO NOT commit unless explicitly asked

## For Machine Learning Tasks
1. **Model Training**:
   - Cross-validation completed
   - Metrics calculated and logged
   - Feature importance analyzed
   - OOF predictions saved

2. **Evaluation**:
   - Competition metric calculated
   - Per-fold scores reported
   - Confusion matrix generated
   - Error analysis performed

3. **Reproducibility**:
   - Config file updated
   - Random seeds set
   - Dependencies documented
   - Results timestamped

## Before Sharing Results
1. Ensure all code runs without errors
2. Document any assumptions made
3. Create summary of results in markdown
4. Save all artifacts to results/
5. Clean up temporary files