const RECIPIENTS = ["mitbpatel128@gmail.com"];
const EXPECTED_TOKEN = "replace-with-a-random-secret";

function doPost(e) {
  try {
    if (!e || !e.postData || !e.postData.contents) {
      return jsonResponse({ ok: false, error: "missing_post_body" });
    }

    const token = (e.parameter && e.parameter.token) || "";
    if (EXPECTED_TOKEN && token !== EXPECTED_TOKEN) {
      return jsonResponse({ ok: false, error: "invalid_token" });
    }

    const payload = JSON.parse(e.postData.contents);
    const subject = `[Market Sentiment] AutoML feature ranking ready: ${payload.run_id || "unknown-run"}`;
    const lines = [
      payload.text || "AutoML feature ranking is ready for review.",
      "",
      `Run ID: ${payload.run_id || "n/a"}`,
      `Candidate feature version: ${payload.candidate_feature_version || "n/a"}`,
      `Candidate feature count: ${payload.candidate_feature_count || "n/a"}`,
      `Selected feature count: ${payload.selected_feature_count || "n/a"}`,
      `Review status: ${payload.review_status || "n/a"}`,
      `Execution status: ${payload.execution_status || "n/a"}`,
      `Report URI: ${payload.report_uri || "n/a"}`,
      `Contract URI: ${payload.contract_uri || "n/a"}`,
      "",
      "Selected features:",
      ...((payload.selected_feature_names || []).map((name) => `- ${name}`)),
    ];

    MailApp.sendEmail({
      to: RECIPIENTS.join(","),
      subject,
      body: lines.join("\n"),
    });

    return jsonResponse({ ok: true, sent_to: RECIPIENTS });
  } catch (error) {
    return jsonResponse({
      ok: false,
      error: String(error),
    });
  }
}

function jsonResponse(payload) {
  return ContentService.createTextOutput(JSON.stringify(payload)).setMimeType(
    ContentService.MimeType.JSON,
  );
}
