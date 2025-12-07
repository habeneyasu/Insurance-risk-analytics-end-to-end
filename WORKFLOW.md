# Development Workflow

This project follows a **Pull Request (PR) based workflow** to ensure code quality and proper review processes.

## Branch Strategy

### Main Branch
- `main` - Production-ready code
- **Never commit directly to main**
- All changes must come through Pull Requests

### Task Branches
- `task-1`, `task-2`, `task-3`, etc. - Feature branches for specific tasks
- Branch naming: `task-{number}` or `feature/{description}`

## Workflow Steps

### 1. Create a Task Branch

```bash
# Start from updated main branch
git checkout main
git pull origin main

# Create and switch to new task branch
git checkout -b task-{number}

# Or for features
git checkout -b feature/{description}
```

### 2. Make Changes

```bash
# Make your code changes
# Test locally
python src/run_eda.py

# Commit changes with descriptive messages
git add .
git commit -m "Task X: Description of changes"
```

### 3. Push Branch to Remote

```bash
# Push branch to remote
git push -u origin task-{number}
```

### 4. Create Pull Request

1. Go to GitHub repository
2. Click "New Pull Request"
3. Select `task-{number}` → `main`
4. Fill out PR template:
   - Description of changes
   - Task checkbox
   - Testing checklist
5. Request review (if applicable)
6. Click "Create Pull Request"

### 5. Review Process

- Code will be automatically checked by CI/CD
- Wait for review approval
- Address any review comments
- Update PR if needed

### 6. Merge PR

Once approved:
- Click "Merge pull request" on GitHub
- Choose merge strategy (squash and merge recommended)
- Delete branch after merge (optional but recommended)

### 7. Update Local Repository

```bash
# Switch back to main
git checkout main

# Pull latest changes
git pull origin main

# Delete local task branch (optional)
git branch -d task-{number}
```

## Commit Message Guidelines

Follow conventional commit format:

```
Task X: Brief description

- Detailed change 1
- Detailed change 2
- Fixes issue #123
```

Examples:
- `Task 1: Add OOP-based EDA modules`
- `Task 2: Initialize DVC and configure data versioning`
- `docs: Update README with setup instructions`
- `fix: Handle lowercase column names in vehicle analysis`

## PR Checklist

Before creating a PR, ensure:

- [ ] Code follows project style guidelines
- [ ] All tests pass
- [ ] Documentation is updated
- [ ] No merge conflicts
- [ ] PR description is complete
- [ ] Related issues are referenced

## Best Practices

1. **Keep branches focused**: One task/feature per branch
2. **Small, frequent commits**: Easier to review and debug
3. **Descriptive commit messages**: Clear what and why
4. **Update before PR**: Rebase on latest main if needed
5. **Test before PR**: Ensure code runs locally
6. **Review your own PR**: Check diff before requesting review

## Current Task Branches

- `task-1`: Git, GitHub, and EDA ✅
- `task-2`: Data Version Control (DVC) ✅
- `task-3`: A/B Hypothesis Testing (upcoming)
- `task-4`: Machine Learning Models (upcoming)

## Troubleshooting

### Merge Conflicts

```bash
# Update your branch with latest main
git checkout task-{number}
git fetch origin
git rebase origin/main

# Resolve conflicts, then
git add .
git rebase --continue
git push --force-with-lease origin task-{number}
```

### Update PR After Changes

```bash
# Make changes, commit, and push
git add .
git commit -m "Update: Description"
git push origin task-{number}
# PR will automatically update
```

