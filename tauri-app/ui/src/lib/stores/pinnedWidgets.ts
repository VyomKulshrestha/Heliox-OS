import { writable } from "svelte/store";
export type PinnedWidget = {
  title: string;
  color: string;
};
export const pinnedWidgets =
  writable<PinnedWidget[]>([]);