use axum::response::IntoResponse;

use crate::templates::{HtmlTemplate, MNISTTemplate};

pub async fn index() -> impl IntoResponse {
    let template = MNISTTemplate {};
    HtmlTemplate(template)
}