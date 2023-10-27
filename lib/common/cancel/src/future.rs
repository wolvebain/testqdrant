use std::future::Future;

use super::*;

pub async fn on_drop<Task, Fut>(task: Task) -> Result<Fut::Output, Error>
where
    Task: FnOnce(CancellationToken) -> Fut,
    Fut: Future + Send + 'static,
    Fut::Output: Send + 'static,
{
    let cancel = CancellationToken::new();

    let future = task(cancel.child_token());

    let guard = cancel.drop_guard();
    let result = tokio::task::spawn(future).await?;
    guard.disarm();

    Ok(result)
}

/// # Safety
///
/// Future have to be cancel-safe!
pub async fn on_token<Fut>(cancel: CancellationToken, future: Fut) -> Result<Fut::Output, Error>
where
    Fut: Future,
{
    tokio::select! {
        biased;
        _ = cancel.cancelled() => Err(Error::Cancelled),
        output = future => Ok(output),
    }
}
