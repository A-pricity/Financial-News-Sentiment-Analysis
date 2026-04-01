# Crawler System Improvements

## Problems Identified

### 1. No Checkpoint/Resume Support
**Current:** Only saves data at the end (`crawl_all` line 108-113)
**Risk:** Hours of crawling lost on failure
**Solution:** 
- Save checkpoint every N articles (configurable)
- Track crawled URLs in checkpoint file
- Resume from last checkpoint on restart

### 2. Serial Execution (No Parallelism)
**Current:** `for source_name in sources:` loops sequentially
**Impact:** 7 sources × avg time = 7× slower
**Solution:**
- Use `concurrent.futures.ThreadPoolExecutor` for parallel crawling
- Each crawler runs in separate thread
- Aggregate results with thread-safe queue

### 3. No Log File
**Current:** Only `logging.basicConfig()` without FileHandler
**Impact:** 
- Terminal cluttered with all messages
- No historical logs for debugging
- Can't track errors after session ends
**Solution:**
- Add RotatingFileHandler
- Separate log levels: INFO to file, WARNING+ to terminal
- Log file: `logs/crawler_YYYYMMDD.log`

## Implementation Plan

### Phase 1: Checkpoint System
- Add `checkpoint_interval` config (default: 50 articles)
- Save checkpoint JSON every N articles
- Track: `{source, url, title, timestamp}`
- On startup: load checkpoint, skip already crawled

### Phase 2: Parallel Crawling  
- Use ThreadPoolExecutor with max_workers config
- Each crawler: independent thread
- Results collected via thread-safe queue
- Progress bar shows all sources simultaneously

### Phase 3: Logging System
- Create `logs/` directory
- RotatingFileHandler: 10MB per file, keep 5
- Terminal: only WARNING+ and progress
- File: DEBUG level for full traceability

## Config Changes

```yaml
crawler:
  # ... existing ...
  checkpoint_interval: 50      # Save every 50 articles
  checkpoint_dir: "checkpoints"
  max_workers: 4               # Parallel threads
  log_dir: "logs"
  log_level: "INFO"
```

## Expected Benefits

| Metric | Before | After |
|--------|--------|-------|
| Data safety | 0% (lose all on fail) | 98%+ (resume from checkpoint) |
| Speed | 1× serial | ~3-4× parallel |
| Debugging | Terminal only | Full log history |
