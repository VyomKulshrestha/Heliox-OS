// The daemon's model_catalog answer as tests see it: a cut of pilot.models.catalog.catalog() (13 November 2026).
import type { ModelCatalog } from "../stores/modelCatalog";

export const CATALOG: ModelCatalog = {
  read_on: "2026-11-13",
  providers: {
    claude: {
      planner_default: "claude-opus-5-5",
      talk_default: "claude-haiku-5-5",
      offered: ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"],
      withdrawn: {
        "claude-3-7-sonnet": { state: "retired", successor: "claude-sonnet-5-5", date: "2026-02-19", says: "" },
      },
    },
    openai: {
      planner_default: "gpt-6-luna",
      talk_default: "gpt-6-luna",
      planner_note:
        "Default chosen from OpenAI's published list on 2026-11-13; not yet checked against the live service by Heliox's tests.",
      talk_note:
        "Default chosen from OpenAI's published list on 2026-11-13; not yet checked against the live service by Heliox's tests.",
      offered: ["gpt-6.1-sol", "gpt-6-luna", "gpt-5.6-luna"],
      withdrawn: {
        "gpt-4o": { state: "deprecated", successor: "gpt-6-luna", date: "2027-01-14", says: "" },
        "gpt-4o-mini": { state: "deprecated", successor: "gpt-6-luna", date: "2027-01-14", says: "" },
        o1: { state: "retired", successor: "gpt-5.6-sol", date: "2026-10-23", says: "" },
      },
    },
    gemini: {
      planner_default: "gemini-3.5-flash-lite",
      talk_default: "gemini-2.5-flash-lite",
      offered: ["gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-2.5-flash-lite"],
      withdrawn: {
        "gemini-2.5-flash": { state: "deprecated", successor: "gemini-3.5-flash-lite", date: "2026-12-01", says: "" },
      },
    },
    openrouter: {
      planner_default: "openrouter/auto",
      talk_default: "openai/gpt-6-luna",
      offered: ["openrouter/auto", "openai/gpt-6-luna"],
      withdrawn: {},
    },
    meta: {
      planner_default: "muse-spark-1.3",
      talk_default: "muse-spark-1.3",
      offered: ["muse-spark-1.3", "muse-spark-1.1"],
      withdrawn: {},
    },
  },
};
