export interface ScrollMetrics {
  scrollTop: number;
  clientHeight: number;
  scrollHeight: number;
}

export const BOTTOM_THRESHOLD_PX = 8;
export const UPWARD_SCROLL_TOLERANCE_PX = 1;

export function isNearBottom(
  metrics: ScrollMetrics,
  threshold = BOTTOM_THRESHOLD_PX,
): boolean {
  return metrics.scrollTop + metrics.clientHeight >= metrics.scrollHeight - threshold;
}

export function movedUpward(
  previousScrollTop: number,
  currentScrollTop: number,
  tolerance = UPWARD_SCROLL_TOLERANCE_PX,
): boolean {
  return currentScrollTop < previousScrollTop - tolerance;
}

export function shouldFollowLatest(atBottom: boolean): boolean {
  return atBottom;
}
