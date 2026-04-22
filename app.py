from __future__ import annotations

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from data_utils import (
    build_missing_values_table,
    build_summary,
    get_datetime_columns,
    get_columns_with_missing_values,
    get_iqr_outlier_summary,
    load_csv,
)
from eda_utils import (
    build_categorical_bar_figure,
    build_correlation_figure,
    build_distribution_figure,
    build_numerical_summary,
    build_pca_scatter_figure,
    build_tuning_figure,
    build_validation_figure,
    get_categorical_columns,
    get_categorical_value_counts,
    get_numerical_columns,
    prepare_pca_projection,
)
from model_utils import (
    compute_regime_confidence_history,
    compute_regime_prediction,
    evaluate_time_series_validation,
    predict_regime_for_scenario,
    run_kmeans_clustering,
    select_features_by_correlation,
    select_features_by_variance,
    time_based_split,
    tune_kmeans_clusters,
)
from quant_utils import (
    apply_regime_labels,
    build_confidence_history_figure,
    build_cluster_distribution_figure,
    build_current_regime_outlook_figure,
    build_cumulative_returns_figure,
    build_drawdown_figure,
    build_regime_insights_table,
    build_regime_feature_profile_figure,
    build_markov_regime_forecast_figure,
    build_market_regime_figure,
    build_regime_timeline_figure,
    build_regime_return_contribution_figure,
    build_regime_summary_table,
    build_regime_transition_figure,
    compute_regime_statistics,
    compute_current_regime_outlook,
    compute_markov_regime_forecast,
    compute_cagr,
    get_time_column_options,
    infer_regime_labels,
    prepare_market_regime_plot_data,
    slice_recent_period,
    build_quant_correlation_figure,
    build_returns_histogram,
    build_strategy_comparison_figure,
    build_strategy_exposure_figure,
    build_rolling_volatility_figure,
    compute_cumulative_returns,
    compute_daily_returns,
    compute_drawdown,
    compute_hit_rate,
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_strategy_returns,
    infer_time_column,
    summarize_strategy_by_regime,
)
from ui_utils import (
    activate_landing_mode,
    configure_page,
    render_chart_frame_end,
    render_chart_frame_start,
    render_control_strip_end,
    render_control_strip_start,
    render_landing_cards,
    render_landing_footer_note,
    render_landing_hero,
    render_landing_nav,
    render_landing_preview,
    render_hero_chart_panel_start,
    render_metric_row,
    render_page_header,
    render_panel_end,
    render_panel_start,
    render_section_header,
    render_sidebar_brand,
    render_sidebar_promo,
    render_spacer,
    render_subpanel_end,
    render_subpanel_start,
    render_topbar,
)


def init_session_state() -> None:
    """Initialize persistent session keys."""
    defaults = {
        "raw_dataframe": None,
        "cleaned_dataframe": None,
        "engineered_dataframe": None,
        "modeled_dataframe": None,
        "selected_feature_columns": [],
        "model_metrics": {},
        "missing_strategy": "None",
        "missing_columns": [],
        "outlier_columns": [],
        "remove_outliers": False,
        "feature_source_columns": [],
        "feature_names": ["Returns", "Rolling Mean", "Rolling Volatility", "Momentum"],
        "rolling_window": 5,
        "correlation_anchor": None,
        "correlation_threshold": 0.2,
        "variance_threshold": 0.0,
        "model_features": [],
        "cluster_count": 3,
        "date_column": None,
        "price_column": None,
        "input_feature_columns": [],
        "final_model_feature_set": [],
        "test_split_ratio": 0.2,
        "validation_splits": 4,
        "tuning_cluster_min": 2,
        "tuning_cluster_max": 6,
        "current_page": "Landing",
        "what_if_return": 0.0,
        "what_if_volatility": 0.01,
        "what_if_rolling_mean": 0.0,
        "pending_scroll_top": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def scroll_page_to_top_if_needed() -> None:
    """Scroll the browser viewport to the top after navigation changes."""
    if not st.session_state.get("pending_scroll_top"):
        return
    components.html(
        """
        <script>
            const scrollTop = () => {
                const parentDoc = window.parent.document;
                const mainSection = parentDoc.querySelector('[data-testid="stAppViewContainer"]');
                if (mainSection) {
                    mainSection.scrollTo({ top: 0, behavior: 'instant' });
                }
                window.parent.scrollTo({ top: 0, behavior: 'instant' });
            };
            scrollTop();
            setTimeout(scrollTop, 30);
        </script>
        """,
        height=0,
    )
    st.session_state.pending_scroll_top = False


@st.cache_data(show_spinner=False)
def apply_cleaning_cached(
    dataframe: pd.DataFrame,
    missing_strategy: str,
    missing_columns: tuple[str, ...],
    remove_outliers: bool,
    outlier_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Apply data cleaning for a given configuration."""
    from data_utils import fill_missing_values, remove_iqr_outliers

    cleaned_dataframe = dataframe.copy()
    if missing_strategy != "None" and missing_columns:
        cleaned_dataframe = fill_missing_values(cleaned_dataframe, missing_strategy, list(missing_columns))
    if remove_outliers and outlier_columns:
        cleaned_dataframe = remove_iqr_outliers(cleaned_dataframe, list(outlier_columns))
    return cleaned_dataframe


@st.cache_data(show_spinner=False)
def apply_feature_engineering_cached(
    dataframe: pd.DataFrame,
    source_columns: tuple[str, ...],
    feature_names: tuple[str, ...],
    window_size: int,
) -> pd.DataFrame:
    """Apply feature engineering for a given configuration."""
    from data_utils import add_financial_features

    if not source_columns or not feature_names:
        return dataframe.copy()
    return add_financial_features(dataframe, list(source_columns), list(feature_names), window_size)


@st.cache_data(show_spinner=False)
def run_kmeans_cached(
    dataframe: pd.DataFrame,
    feature_columns: tuple[str, ...],
    n_clusters: int,
    time_column: str | None,
    test_ratio: float,
) -> tuple[pd.DataFrame, float, float | None, int, int, int, float | None]:
    """Run KMeans clustering for a given configuration."""
    result = run_kmeans_clustering(
        dataframe,
        list(feature_columns),
        n_clusters,
        time_column=time_column,
        test_ratio=test_ratio,
    )
    return (
        result.dataframe,
        result.inertia,
        result.silhouette_score,
        result.training_row_count,
        result.testing_row_count,
        result.labeled_row_count,
        result.davies_bouldin_score,
    )


@st.cache_data(show_spinner=False)
def compute_split_cached(
    dataframe: pd.DataFrame,
    time_column: str | None,
    test_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str | None]:
    """Compute a reusable time-based split for the processed dataset."""
    result = time_based_split(dataframe, time_column, test_ratio)
    return result.sorted_dataframe, result.train_dataframe, result.test_dataframe, result.time_column


@st.cache_data(show_spinner=False)
def compute_validation_cached(
    dataframe: pd.DataFrame,
    feature_columns: tuple[str, ...],
    time_column: str | None,
    n_clusters: int,
    n_splits: int,
) -> tuple[pd.DataFrame, float | None, float | None]:
    """Compute TimeSeriesSplit validation metrics."""
    result = evaluate_time_series_validation(
        dataframe,
        list(feature_columns),
        time_column,
        n_clusters,
        n_splits,
    )
    return result.summary, result.mean_silhouette, result.mean_davies_bouldin


@st.cache_data(show_spinner=False)
def compute_tuning_cached(
    dataframe: pd.DataFrame,
    feature_columns: tuple[str, ...],
    time_column: str | None,
    cluster_values: tuple[int, ...],
    n_splits: int,
) -> pd.DataFrame:
    """Compute cluster-count tuning metrics."""
    return tune_kmeans_clusters(
        dataframe,
        list(feature_columns),
        time_column,
        list(cluster_values),
        n_splits,
    )


@st.cache_data(show_spinner=False)
def compute_regime_prediction_cached(
    dataframe: pd.DataFrame,
    feature_columns: tuple[str, ...],
    n_clusters: int,
    time_column: str | None,
    test_ratio: float,
) -> tuple[int | None, int | None, float | None]:
    """Compute the latest regime estimate and nearest alternate regime."""
    result = compute_regime_prediction(
        dataframe,
        list(feature_columns),
        n_clusters,
        time_column,
        test_ratio,
    )
    return result.current_regime, result.predicted_next_regime, result.confidence


@st.cache_data(show_spinner=False)
def compute_what_if_prediction_cached(
    dataframe: pd.DataFrame,
    feature_columns: tuple[str, ...],
    n_clusters: int,
    time_column: str | None,
    test_ratio: float,
    scenario_items: tuple[tuple[str, float], ...],
) -> tuple[int | None, float | None]:
    """Estimate the closest regime for an interactive what-if scenario."""
    scenario_features = {key: value for key, value in scenario_items}
    result = predict_regime_for_scenario(
        dataframe,
        list(feature_columns),
        n_clusters,
        time_column,
        test_ratio,
        scenario_features,
    )
    return result.current_regime, result.confidence


@st.cache_data(show_spinner=False)
def compute_regime_confidence_history_cached(
    dataframe: pd.DataFrame,
    feature_columns: tuple[str, ...],
    n_clusters: int,
    time_column: str | None,
    test_ratio: float,
) -> pd.DataFrame:
    """Compute cached confidence-history series for sparkline rendering."""
    return compute_regime_confidence_history(
        dataframe,
        list(feature_columns),
        n_clusters,
        time_column,
        test_ratio,
    )


@st.cache_data(show_spinner=False)
def compute_regime_semantics_cached(
    dataframe: pd.DataFrame,
    price_column: str,
    time_column: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[tuple[int, str], ...]]:
    """Compute semantic regime mapping and attach regime_type labels."""
    regime_stats = compute_regime_statistics(
        dataframe,
        regime_col="market_regime",
        price_column=price_column,
        time_column=time_column,
    )
    mapping = infer_regime_labels(regime_stats)
    labeled_dataframe = apply_regime_labels(dataframe, mapping, regime_col="market_regime")
    return labeled_dataframe, regime_stats, tuple(sorted(mapping.items()))


def ensure_role_defaults(raw_dataframe: pd.DataFrame) -> None:
    """Seed dataset role selectors from the uploaded dataframe."""
    datetime_columns = get_datetime_columns(raw_dataframe)
    numerical_columns = get_numerical_columns(raw_dataframe)

    if st.session_state.date_column not in raw_dataframe.columns:
        st.session_state.date_column = datetime_columns[0] if datetime_columns else None
    if st.session_state.price_column not in numerical_columns:
        st.session_state.price_column = "Close" if "Close" in numerical_columns else (numerical_columns[0] if numerical_columns else None)
    if not st.session_state.input_feature_columns:
        default_features = [column for column in numerical_columns if column != st.session_state.price_column]
        st.session_state.input_feature_columns = default_features[: min(6, len(default_features))] or numerical_columns[: min(6, len(numerical_columns))]


def render_sidebar() -> str:
    """Render sidebar navigation with grouped sections."""
    pages = [
        ("Home", ["Landing"]),
        ("Data", ["Dataset Overview"]),
        ("Model", ["ML Pipeline"]),
        ("Quantitative Tools", ["Market Regime Analysis", "Trading Strategy", "Correlation Matrix", "Volatility Analysis", "Returns Analysis"]),
        ("System", ["About"]),
    ]
    nav_labels = {
        "Landing": "Overview",
        "Dataset Overview": "Dataset",
        "ML Pipeline": "Model Console",
        "Market Regime Analysis": "Regime Monitor",
        "Trading Strategy": "Strategy Desk",
        "Correlation Matrix": "Market Structure",
        "Volatility Analysis": "Volatility Monitor",
        "Returns Analysis": "Return Monitor",
        "About": "About",
    }

    with st.sidebar:
        render_sidebar_brand()
        render_sidebar_promo(
            "Live Workspace",
            "Move from ingestion to regime monitoring and market structure views without leaving the dashboard rail.",
            "Desk View",
        )
        for group_name, items in pages:
            st.markdown(f'<div class="sidebar-group">{group_name}</div>', unsafe_allow_html=True)
            for item in items:
                is_active = st.session_state.current_page == item
                if is_active:
                    st.markdown('<div class="active-nav">', unsafe_allow_html=True)
                if st.button(nav_labels.get(item, item), key=f"nav_{item}", type="secondary"):
                    st.session_state.current_page = item
                    st.session_state.pending_scroll_top = True
                if is_active:
                    st.markdown("</div>", unsafe_allow_html=True)
        return st.session_state.current_page


def render_landing_page() -> None:
    """Render a premium landing page before entering the dashboard workspace."""
    activate_landing_mode()
    render_landing_nav()
    render_landing_hero()
    st.markdown('<div class="landing-cta-wrap">', unsafe_allow_html=True)
    left_pad, action_primary, action_secondary, right_pad = st.columns([3.1, 1.45, 1.45, 3.1], gap="small")
    with action_primary:
        if st.button("Enter Workspace", key="landing_enter_workspace", type="primary"):
            st.session_state.current_page = "Market Regime Analysis"
            st.session_state.pending_scroll_top = True
            st.rerun()
    with action_secondary:
        if st.button("View Data Setup", key="landing_view_dataset", type="secondary"):
            st.session_state.current_page = "Dataset Overview"
            st.session_state.pending_scroll_top = True
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
    render_landing_preview()
    render_landing_cards()
    render_landing_footer_note()


def ensure_pipeline_defaults(raw_dataframe: pd.DataFrame) -> None:
    """Seed pipeline defaults from the current raw dataset."""
    ensure_role_defaults(raw_dataframe)
    numerical_columns = get_numerical_columns(raw_dataframe)
    if not st.session_state.missing_columns:
        st.session_state.missing_columns = get_columns_with_missing_values(raw_dataframe)
    if not st.session_state.outlier_columns:
        st.session_state.outlier_columns = numerical_columns
    if not st.session_state.feature_source_columns:
        preferred_sources = [
            column for column in st.session_state.input_feature_columns if column in numerical_columns
        ]
        st.session_state.feature_source_columns = preferred_sources[: min(3, len(preferred_sources))] or numerical_columns[: min(2, len(numerical_columns))]
    if st.session_state.correlation_anchor is None and numerical_columns:
        st.session_state.correlation_anchor = (
            st.session_state.price_column if st.session_state.price_column in numerical_columns else numerical_columns[0]
        )


def compute_pipeline_outputs() -> None:
    """Compute the pipeline outputs from current session-state controls."""
    raw_dataframe = st.session_state.raw_dataframe
    if raw_dataframe is None:
        return

    ensure_pipeline_defaults(raw_dataframe)

    cleaned_dataframe = apply_cleaning_cached(
        raw_dataframe,
        st.session_state.missing_strategy,
        tuple(st.session_state.missing_columns),
        st.session_state.remove_outliers,
        tuple(st.session_state.outlier_columns),
    )
    engineered_dataframe = apply_feature_engineering_cached(
        cleaned_dataframe,
        tuple(st.session_state.feature_source_columns),
        tuple(st.session_state.feature_names),
        st.session_state.rolling_window,
    )

    split_sorted, train_dataframe, test_dataframe, resolved_time_column = compute_split_cached(
        engineered_dataframe,
        st.session_state.date_column,
        st.session_state.test_split_ratio,
    )
    engineered_dataframe = split_sorted
    engineered_numerical = get_numerical_columns(engineered_dataframe)
    if st.session_state.correlation_anchor not in engineered_numerical and engineered_numerical:
        st.session_state.correlation_anchor = (
            st.session_state.price_column if st.session_state.price_column in engineered_numerical else engineered_numerical[0]
        )

    candidate_columns = [
        column
        for column in st.session_state.input_feature_columns
        if column in engineered_numerical
    ] or engineered_numerical

    correlation_selected, correlation_table = select_features_by_correlation(
        engineered_dataframe,
        candidate_columns,
        st.session_state.correlation_anchor,
        st.session_state.correlation_threshold,
    )
    variance_selected, variance_table = select_features_by_variance(
        engineered_dataframe,
        correlation_selected,
        st.session_state.variance_threshold,
    )

    selected_feature_columns = variance_selected or correlation_selected or engineered_numerical
    st.session_state.selected_feature_columns = selected_feature_columns
    if st.session_state.final_model_feature_set:
        selected_feature_columns = [
            column for column in st.session_state.final_model_feature_set if column in selected_feature_columns
        ] or selected_feature_columns
    if not st.session_state.model_features:
        st.session_state.model_features = selected_feature_columns[: min(3, len(selected_feature_columns))]
    st.session_state.final_model_feature_set = selected_feature_columns

    model_features = [
        column for column in st.session_state.model_features if column in selected_feature_columns
    ]
    modeled_dataframe = engineered_dataframe.copy()
    inertia = None
    silhouette = None
    training_rows = 0
    testing_rows = 0
    labeled_rows = 0
    davies_bouldin = None

    if len(model_features) >= 2:
        max_clusters = max(2, min(10, len(train_dataframe)))
        st.session_state.cluster_count = 3 if max_clusters >= 3 else max_clusters
        try:
            (
                modeled_dataframe,
                inertia,
                silhouette,
                training_rows,
                testing_rows,
                labeled_rows,
                davies_bouldin,
            ) = run_kmeans_cached(
                engineered_dataframe,
                tuple(model_features),
                st.session_state.cluster_count,
                resolved_time_column,
                st.session_state.test_split_ratio,
            )
        except ValueError:
            modeled_dataframe = engineered_dataframe.copy()

    validation_summary = pd.DataFrame()
    validation_silhouette = None
    validation_davies_bouldin = None
    if len(model_features) >= 2:
        (
            validation_summary,
            validation_silhouette,
            validation_davies_bouldin,
        ) = compute_validation_cached(
            engineered_dataframe,
            tuple(model_features),
            resolved_time_column,
            st.session_state.cluster_count,
            st.session_state.validation_splits,
        )

    tuning_summary = pd.DataFrame()
    if len(model_features) >= 2:
        cluster_min = min(st.session_state.tuning_cluster_min, st.session_state.tuning_cluster_max)
        cluster_max = max(st.session_state.tuning_cluster_min, st.session_state.tuning_cluster_max)
        cluster_max = min(cluster_max, max(2, min(10, len(train_dataframe))))
        cluster_min = min(cluster_min, cluster_max)
        cluster_values = tuple(range(cluster_min, cluster_max + 1))
        tuning_summary = compute_tuning_cached(
            engineered_dataframe,
            tuple(model_features),
            resolved_time_column,
            cluster_values,
            st.session_state.validation_splits,
        )

    st.session_state.cleaned_dataframe = cleaned_dataframe
    st.session_state.engineered_dataframe = engineered_dataframe
    st.session_state.modeled_dataframe = modeled_dataframe
    st.session_state.model_metrics = {
        "inertia": inertia,
        "silhouette_score": silhouette,
        "davies_bouldin_score": davies_bouldin,
        "correlation_table": correlation_table,
        "variance_table": variance_table,
        "model_features": model_features,
        "time_column": resolved_time_column,
        "train_dataframe": train_dataframe,
        "test_dataframe": test_dataframe,
        "train_rows": len(train_dataframe),
        "test_rows": len(test_dataframe),
        "training_row_count": training_rows,
        "testing_row_count": testing_rows,
        "labeled_row_count": labeled_rows,
        "validation_summary": validation_summary,
        "validation_mean_silhouette": validation_silhouette,
        "validation_mean_davies_bouldin": validation_davies_bouldin,
        "tuning_summary": tuning_summary,
    }


def get_pipeline_insights(engineered_dataframe: pd.DataFrame, feature_columns: list[str]) -> list[str]:
    """Build concise pipeline insight strings for the EDA section."""
    insights: list[str] = []
    if not feature_columns:
        return ["Select at least two numerical features to generate clustering-ready diagnostics."]

    missing_cells = int(engineered_dataframe[feature_columns].isna().sum().sum())
    insights.append(f"The selected modeling feature set currently spans {len(feature_columns)} columns.")
    insights.append(f"The engineered feature frame contains {missing_cells:,} missing cells across the selected features.")

    if st.session_state.price_column and st.session_state.price_column in engineered_dataframe.columns:
        price_series = pd.to_numeric(engineered_dataframe[st.session_state.price_column], errors="coerce").dropna()
        if len(price_series) >= 2:
            insights.append(
                f"The selected price series ranges from {price_series.min():.2f} to {price_series.max():.2f} across the processed timeline."
            )
    return insights


def get_feature_quality_warnings() -> list[str]:
    """Build warning messages for weak clustering feature configurations."""
    warnings: list[str] = []
    metrics = st.session_state.model_metrics
    model_features = metrics.get("model_features", [])
    training_rows = int(metrics.get("training_row_count", 0))
    silhouette = metrics.get("silhouette_score")
    variance_table = metrics.get("variance_table", pd.DataFrame())

    if len(model_features) < 2:
        warnings.append("Select at least two final modeling features so clustering can separate regimes meaningfully.")
    if training_rows and training_rows < 100:
        warnings.append("The valid training sample is fairly small after feature completeness filtering, so regime stability may be weak.")
    if not variance_table.empty and "Variance" in variance_table.columns and (variance_table["Variance"] <= 1e-6).any():
        warnings.append("One or more candidate features have near-zero variance, which can weaken clustering signal.")
    if silhouette is not None and silhouette < 0.2:
        warnings.append("The current silhouette score is low, which can indicate weak separation or too many/too few clusters.")
    return warnings


def get_feature_profile_columns(dataframe: pd.DataFrame) -> list[str]:
    """Choose more interpretable, less redundant columns for the regime feature profile."""
    candidate_columns = (
        st.session_state.model_metrics.get("model_features", [])
        or st.session_state.final_model_feature_set
        or st.session_state.selected_feature_columns
    )
    candidate_columns = [column for column in candidate_columns if column in dataframe.columns]
    if not candidate_columns:
        return []

    engineered_priority = [
        column
        for column in candidate_columns
        if any(token in column.lower() for token in ("returns", "volatility", "rolling", "spread", "range"))
    ]
    profile_columns = engineered_priority if len(engineered_priority) >= 2 else candidate_columns

    if len(profile_columns) > 1:
        numeric_frame = dataframe[profile_columns].apply(pd.to_numeric, errors="coerce")
        correlation_matrix = numeric_frame.corr(numeric_only=True).abs()
        deduplicated_columns: list[str] = []
        for column in profile_columns:
            if not deduplicated_columns:
                deduplicated_columns.append(column)
                continue
            if correlation_matrix.loc[column, deduplicated_columns].fillna(0).max() < 0.95:
                deduplicated_columns.append(column)
        if len(deduplicated_columns) >= 2:
            profile_columns = deduplicated_columns

    return profile_columns[: min(6, len(profile_columns))]


def render_pipeline_summary() -> None:
    """Render a concise academic pipeline summary block."""
    metrics = st.session_state.model_metrics
    modeled_dataframe = st.session_state.modeled_dataframe
    regime_count = 0
    if modeled_dataframe is not None and "market_regime" in modeled_dataframe.columns:
        regime_count = int(modeled_dataframe["market_regime"].dropna().nunique())

    render_panel_start(compact=True)
    render_section_header(
        "Pipeline Summary",
        "A compact academic summary of the current dataset state, feature space, chronological split, and clustering output.",
    )
    render_metric_row(
        [
            ("Input Rows", f"{len(st.session_state.raw_dataframe):,}", "Original uploaded observations"),
            ("Cleaned Rows", f"{len(st.session_state.cleaned_dataframe):,}", "Rows after cleaning configuration"),
            ("Engineered Features", f"{len(st.session_state.engineered_dataframe.columns):,}", "Columns after feature engineering"),
            ("Final Features", f"{len(metrics.get('model_features', [])):,}", "Features used by KMeans"),
            ("Train/Test", f"{metrics.get('train_rows', 0):,} / {metrics.get('test_rows', 0):,}", "Chronological split"),
            ("Clusters", f"{regime_count:,}", "Detected market regimes"),
        ]
    )
    warning_messages = get_feature_quality_warnings()
    if warning_messages:
        render_spacer()
        render_subpanel_start()
        for message in warning_messages:
            st.warning(message)
        render_subpanel_end()
    render_panel_end()


def render_dataset_overview_page() -> None:
    """Render the dataset overview page."""
    render_page_header(
        "Data",
        "Dataset Overview",
        "Upload a market dataset once and persist it across the full analytics workspace.",
    )

    render_panel_start()
    render_section_header(
        "Dataset Ingestion",
        "Load a single market dataset and persist it across the analytics workflow.",
    )
    uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"], label_visibility="collapsed")
    if uploaded_file is not None:
        st.session_state.raw_dataframe = load_csv(uploaded_file)

    raw_dataframe = st.session_state.raw_dataframe
    if raw_dataframe is None:
        st.info("Upload a CSV file to begin the workflow.")
        render_panel_end()
        return
    render_panel_end()

    ensure_role_defaults(raw_dataframe)
    summary = build_summary(raw_dataframe)
    render_metric_row(
        [
            ("Rows", f"{summary.rows:,}", "Total observations loaded"),
            ("Columns", f"{summary.columns_count:,}", "Available fields in the dataset"),
            ("Missing Values", f"{int(raw_dataframe.isna().sum().sum()):,}", "Incomplete cells requiring review"),
            ("Numeric Features", f"{len(get_numerical_columns(raw_dataframe)):,}", "Fields eligible for quantitative analysis"),
        ]
    )
    render_spacer("xl")

    datetime_columns = get_datetime_columns(raw_dataframe)
    numerical_columns = get_numerical_columns(raw_dataframe)
    default_feature_columns = [
        column for column in st.session_state.input_feature_columns if column in numerical_columns
    ] or [column for column in numerical_columns if column != st.session_state.price_column][: min(6, len(numerical_columns))]

    role_column, pca_column = st.columns([1.05, 1.35], gap="large")
    with role_column:
        render_panel_start()
        render_section_header(
            "Column Roles",
            "Assign the time field, primary market price, and candidate features used throughout the pipeline.",
        )
        time_options = ["None"] + datetime_columns
        date_index = time_options.index(st.session_state.date_column) if st.session_state.date_column in time_options else 0
        selected_date = st.selectbox("Date column", options=time_options, index=date_index)
        st.session_state.date_column = None if selected_date == "None" else selected_date

        if numerical_columns:
            default_price = st.session_state.price_column if st.session_state.price_column in numerical_columns else ("Close" if "Close" in numerical_columns else numerical_columns[0])
            st.session_state.price_column = st.selectbox(
                "Price column",
                options=numerical_columns,
                index=numerical_columns.index(default_price),
            )
            st.session_state.input_feature_columns = st.multiselect(
                "Feature columns",
                options=numerical_columns,
                default=default_feature_columns,
            )
        else:
            st.warning("No numerical columns are available for feature selection.")

        if st.session_state.date_column:
            sorted_preview = raw_dataframe.sort_values(st.session_state.date_column)
            start_date = sorted_preview[st.session_state.date_column].min()
            end_date = sorted_preview[st.session_state.date_column].max()
            st.caption(f"Detected time span: {start_date} to {end_date}")
        compute_pipeline_outputs()
        export_dataframe = st.session_state.modeled_dataframe if st.session_state.modeled_dataframe is not None else raw_dataframe
        st.download_button(
            "Download Processed Dataset",
            data=export_dataframe.to_csv(index=False).encode("utf-8"),
            file_name="market_regime_processed_dataset.csv",
            mime="text/csv",
            use_container_width=True,
        )
        render_panel_end()

    with pca_column:
        render_panel_start()
        render_section_header(
            "PCA Projection",
            "A compact two-dimensional view of the selected feature space to satisfy the overall data-shape requirement.",
        )
        pca_feature_columns = [
            column for column in st.session_state.input_feature_columns if column in numerical_columns
        ]
        pca_source = st.session_state.modeled_dataframe if st.session_state.modeled_dataframe is not None else raw_dataframe
        pca_result = prepare_pca_projection(pca_source, pca_feature_columns)
        if pca_result.error:
            st.warning(pca_result.error)
        else:
            render_metric_row(
                [
                    (
                        "PC1 Variance",
                        f"{pca_result.explained_variance_ratio[0] * 100:.1f}%",
                        "Variance explained by the first principal component",
                    ),
                    (
                        "PC2 Variance",
                        f"{pca_result.explained_variance_ratio[1] * 100:.1f}%",
                        "Variance explained by the second principal component",
                    ),
                    (
                        "Plotted Rows",
                        f"{pca_result.plotted_rows:,} / {pca_result.total_rows:,}",
                        "Sampled plotted rows versus complete rows used for PCA",
                    ),
                    (
                        "Tail Clipping",
                        f"{pca_result.clipped_feature_count:,}",
                        "Features gently clipped at 1st/99th percentiles for cleaner visualization",
                    ),
                ]
            )
            render_spacer()
            render_chart_frame_start()
            st.plotly_chart(
                build_pca_scatter_figure(pca_source, pca_feature_columns),
                use_container_width=True,
                key="dataset_overview_pca_scatter",
            )
            render_chart_frame_end()
            st.caption(
                "This PCA view is used for feature-space visualization only. Extreme tails are lightly clipped before projection so the point cloud stays readable, and regimes are used as colors when available."
            )
        render_panel_end()

    render_spacer("xl")
    preview_column, info_column = st.columns([1.55, 1.0], gap="large")
    with preview_column:
        render_panel_start()
        render_section_header(
            "Dataset Preview",
            "A compact view of the uploaded dataset so structure and formatting can be checked quickly.",
        )
        st.dataframe(raw_dataframe.head(15), use_container_width=True, height=430)
        render_panel_end()
    with info_column:
        render_panel_start()
        render_section_header(
            "Schema",
            "Field inventory and missing-value summary kept secondary to the main data preview.",
        )
        columns_tab, missing_tab = st.tabs(["Columns", "Missing Values"])
        with columns_tab:
            render_subpanel_start()
            st.dataframe(
                pd.DataFrame({"Column": summary.columns}),
                use_container_width=True,
                hide_index=True,
                height=250,
            )
            render_subpanel_end()
        with missing_tab:
            render_subpanel_start()
            st.dataframe(
                build_missing_values_table(raw_dataframe),
                use_container_width=True,
                hide_index=True,
                height=250,
            )
            render_subpanel_end()
        render_panel_end()


def render_pipeline_eda(engineered_dataframe: pd.DataFrame) -> None:
    """Render EDA content."""
    with st.expander("Exploratory Data Analysis", expanded=True):
        render_section_header(
            "Exploratory Data Analysis",
            "Use summary diagnostics and chart-first views to inspect the processed feature space before clustering.",
        )
        numerical_columns = get_numerical_columns(engineered_dataframe)
        categorical_columns = get_categorical_columns(engineered_dataframe)
        insights = get_pipeline_insights(engineered_dataframe, st.session_state.selected_feature_columns)

        render_subpanel_start()
        for insight in insights:
            st.write(f"- {insight}")
        render_subpanel_end()
        render_spacer()

        left_column, right_column = st.columns([1.05, 1.45], gap="large")
        with left_column:
            render_subpanel_start()
            st.dataframe(
                build_numerical_summary(
                    engineered_dataframe,
                    numerical_columns[: min(8, len(numerical_columns))],
                ),
                use_container_width=True,
                hide_index=True,
                height=320,
            )
            render_subpanel_end()
        with right_column:
            if len(numerical_columns) >= 2:
                corr_columns = st.multiselect(
                    "Correlation features",
                    options=numerical_columns,
                    default=numerical_columns[: min(6, len(numerical_columns))],
                    key="pipeline_corr_features",
                )
                if len(corr_columns) >= 2:
                    render_chart_frame_start()
                    st.plotly_chart(
                        build_correlation_figure(engineered_dataframe, corr_columns),
                        use_container_width=True,
                        key="pipeline_eda_correlation",
                    )
                    render_chart_frame_end()

        if numerical_columns:
            render_spacer()
            distribution_column = st.selectbox(
                "Distribution column",
                options=numerical_columns,
                key="pipeline_distribution_column",
            )
            render_chart_frame_start()
            st.plotly_chart(
                build_distribution_figure(engineered_dataframe, distribution_column),
                use_container_width=True,
                key="pipeline_eda_distribution",
            )
            render_chart_frame_end()

        if categorical_columns:
            render_spacer()
            categorical_column = st.selectbox(
                "Categorical column",
                options=categorical_columns,
                key="pipeline_categorical_column",
            )
            top_n = st.slider(
                "Top categories",
                min_value=5,
                max_value=20,
                value=10,
                key="pipeline_top_n",
            )
            chart_column, table_column = st.columns([1.4, 1.0], gap="large")
            with chart_column:
                render_chart_frame_start()
                st.plotly_chart(
                    build_categorical_bar_figure(engineered_dataframe, categorical_column, top_n),
                    use_container_width=True,
                    key="pipeline_eda_categorical_bar",
                )
                render_chart_frame_end()
            with table_column:
                render_subpanel_start()
                st.dataframe(
                    get_categorical_value_counts(engineered_dataframe, categorical_column, top_n),
                    use_container_width=True,
                    hide_index=True,
                    height=360,
                )
                render_subpanel_end()


def render_pipeline_cleaning(raw_dataframe: pd.DataFrame) -> None:
    """Render cleaning controls."""
    with st.expander("Data Cleaning", expanded=True):
        render_section_header(
            "Data Cleaning",
            "Control missing-value treatment and IQR-based outlier handling inside a cleaner analysis console layout.",
        )
        numerical_columns = get_numerical_columns(raw_dataframe)
        missing_columns = get_columns_with_missing_values(raw_dataframe)
        cleaned_dataframe = st.session_state.cleaned_dataframe

        render_metric_row(
            [
                ("Rows Before", f"{len(raw_dataframe):,}", "Original uploaded observations"),
                ("Rows After", f"{len(cleaned_dataframe):,}" if cleaned_dataframe is not None else "N/A", "Rows after cleaning configuration"),
                (
                    "Dropped Rows",
                    f"{max(len(raw_dataframe) - len(cleaned_dataframe), 0):,}" if cleaned_dataframe is not None else "N/A",
                    "Rows removed mainly through optional outlier filtering",
                ),
            ]
        )
        render_spacer()

        controls_column, summary_column = st.columns([1.1, 1.0], gap="large")
        with controls_column:
            st.session_state.missing_strategy = st.selectbox(
                "Missing value strategy",
                options=["None", "Mean", "Median", "Mode"],
                index=["None", "Mean", "Median", "Mode"].index(st.session_state.missing_strategy),
            )
            st.session_state.missing_columns = st.multiselect(
                "Columns to clean",
                options=missing_columns,
                default=[
                    column for column in st.session_state.missing_columns if column in missing_columns
                ]
                or missing_columns,
            )
            st.session_state.outlier_columns = st.multiselect(
                "Columns for IQR outlier detection",
                options=numerical_columns,
                default=[
                    column for column in st.session_state.outlier_columns if column in numerical_columns
                ]
                or numerical_columns,
            )
            st.session_state.remove_outliers = st.checkbox(
                "Remove IQR outliers",
                value=st.session_state.remove_outliers,
            )

        with summary_column:
            render_subpanel_start()
            st.dataframe(
                get_iqr_outlier_summary(raw_dataframe, st.session_state.outlier_columns),
                use_container_width=True,
                hide_index=True,
                height=320,
            )
            render_subpanel_end()


def render_pipeline_feature_engineering(cleaned_dataframe: pd.DataFrame) -> None:
    """Render feature-engineering controls."""
    with st.expander("Feature Engineering", expanded=True):
        render_section_header(
            "Feature Engineering",
            "Create return, rolling mean, and rolling volatility features while keeping controls compact and modular.",
        )
        numerical_columns = get_numerical_columns(cleaned_dataframe)
        controls_column, results_column = st.columns([1.15, 0.95], gap="large")
        with controls_column:
            st.session_state.feature_source_columns = st.multiselect(
                "Source columns",
                options=numerical_columns,
                default=[
                    column for column in st.session_state.feature_source_columns if column in numerical_columns
                ]
                or numerical_columns[: min(2, len(numerical_columns))],
            )
            st.session_state.feature_names = st.multiselect(
                "Financial features",
                options=["Returns", "Rolling Mean", "Rolling Volatility", "Momentum", "Volume Change"],
                default=st.session_state.feature_names,
            )
            st.session_state.rolling_window = st.slider(
                "Rolling window",
                min_value=2,
                max_value=60,
                value=st.session_state.rolling_window,
            )

        with results_column:
            engineered_dataframe = st.session_state.engineered_dataframe
            if engineered_dataframe is not None:
                added_columns = [
                    column for column in engineered_dataframe.columns if column not in cleaned_dataframe.columns
                ]
                render_subpanel_start()
                st.caption("Rolling features and percentage returns can introduce leading NaNs; clustering later uses only complete rows.")
                st.dataframe(
                    pd.DataFrame({"Engineered Columns": added_columns}),
                    use_container_width=True,
                    hide_index=True,
                    height=270,
                )
                render_subpanel_end()


def render_pipeline_feature_selection(engineered_dataframe: pd.DataFrame) -> None:
    """Render feature-selection controls."""
    with st.expander("Feature Selection", expanded=True):
        render_section_header(
            "Feature Selection",
            "Narrow the factor universe using correlation relevance and variance thresholds before clustering.",
        )
        numerical_columns = get_numerical_columns(engineered_dataframe)
        if not numerical_columns:
            st.info("No numerical features are available for selection.")
            return

        if st.session_state.correlation_anchor not in numerical_columns:
            st.session_state.correlation_anchor = numerical_columns[0]

        controls_column, tables_column = st.columns([1.0, 1.35], gap="large")
        with controls_column:
            st.session_state.correlation_anchor = st.selectbox(
                "Correlation anchor",
                options=numerical_columns,
                index=numerical_columns.index(st.session_state.correlation_anchor),
            )
            st.session_state.correlation_threshold = st.slider(
                "Correlation threshold",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.correlation_threshold),
                step=0.05,
            )
            st.session_state.variance_threshold = st.slider(
                "Variance threshold",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.variance_threshold),
                step=0.01,
            )
            final_feature_candidates = st.session_state.selected_feature_columns or numerical_columns
            st.session_state.final_model_feature_set = st.multiselect(
                "Final modeling feature set",
                options=final_feature_candidates,
                default=[
                    column
                    for column in st.session_state.final_model_feature_set
                    if column in final_feature_candidates
                ] or final_feature_candidates[: min(6, len(final_feature_candidates))],
            )

        correlation_table = st.session_state.model_metrics.get("correlation_table", pd.DataFrame())
        variance_table = st.session_state.model_metrics.get("variance_table", pd.DataFrame())
        with tables_column:
            left, right = st.columns(2, gap="large")
            with left:
                st.caption("Correlation Ranking")
                render_subpanel_start()
                st.dataframe(correlation_table, use_container_width=True, hide_index=True, height=270)
                render_subpanel_end()
            with right:
                st.caption("Variance Ranking")
                render_subpanel_start()
                st.dataframe(variance_table, use_container_width=True, hide_index=True, height=270)
                render_subpanel_end()


def render_pipeline_problem_setup() -> None:
    """Render the problem framing block for the academic pipeline."""
    render_panel_start()
    render_section_header(
        "Problem Setup",
        "Frame the project explicitly as an unsupervised market-regime workflow adapted for financial time-series data.",
    )
    render_metric_row(
        [
            ("Problem Type", "Unsupervised Learning", "No explicit target labels are required for clustering"),
            ("Task", "Market Regime Detection", "Group market states based on common factor behavior"),
            ("Model Family", "Clustering", "KMeans remains the primary supported model"),
        ]
    )
    render_panel_end()


def render_pipeline_data_split(engineered_dataframe: pd.DataFrame) -> None:
    """Render the time-based split configuration and summary."""
    with st.expander("Data Split", expanded=True):
        render_section_header(
            "Data Split",
            "Use an earlier-to-later chronological split rather than a random split so evaluation respects market ordering.",
        )
        st.session_state.test_split_ratio = st.slider(
            "Evaluation split percentage",
            min_value=0.1,
            max_value=0.4,
            value=float(st.session_state.test_split_ratio),
            step=0.05,
            format="%.2f",
        )
        split_metrics = st.session_state.model_metrics
        train_dataframe = split_metrics.get("train_dataframe", engineered_dataframe.iloc[0:0])
        test_dataframe = split_metrics.get("test_dataframe", engineered_dataframe.iloc[0:0])
        resolved_time_column = split_metrics.get("time_column")

        render_metric_row(
            [
                ("Train Rows", f"{len(train_dataframe):,}", "Earlier observations used for model fitting"),
                ("Test Rows", f"{len(test_dataframe):,}", "Later observations reserved for evaluation"),
                ("Test Share", f"{st.session_state.test_split_ratio:.0%}", "Chronological evaluation holdout"),
            ]
        )
        if resolved_time_column and not train_dataframe.empty and not test_dataframe.empty:
            render_spacer()
            render_subpanel_start()
            st.write(
                f"Training period: {train_dataframe[resolved_time_column].min()} to {train_dataframe[resolved_time_column].max()}"
            )
            st.write(
                f"Testing period: {test_dataframe[resolved_time_column].min()} to {test_dataframe[resolved_time_column].max()}"
            )
            render_subpanel_end()


def render_pipeline_validation(engineered_dataframe: pd.DataFrame) -> None:
    """Render validation strategy and cross-split diagnostics."""
    with st.expander("Validation Strategy", expanded=True):
        render_section_header(
            "Validation Strategy",
            "TimeSeriesSplit replaces ordinary K-Fold so each validation fold preserves the temporal ordering of market observations.",
        )
        st.session_state.validation_splits = st.slider(
            "TimeSeriesSplit folds",
            min_value=2,
            max_value=8,
            value=int(st.session_state.validation_splits),
        )
        validation_summary = st.session_state.model_metrics.get("validation_summary", pd.DataFrame())
        mean_silhouette = st.session_state.model_metrics.get("validation_mean_silhouette")
        mean_db = st.session_state.model_metrics.get("validation_mean_davies_bouldin")
        render_metric_row(
            [
                ("Validation Folds", f"{len(validation_summary):,}", "Completed chronological validation windows"),
                ("Mean Silhouette", f"{mean_silhouette:.3f}" if mean_silhouette is not None else "N/A", "Average separation across folds"),
                ("Mean Davies-Bouldin", f"{mean_db:.3f}" if mean_db is not None else "N/A", "Lower values suggest tighter clusters"),
            ]
        )
        render_spacer()
        left_column, right_column = st.columns([1.3, 1.0], gap="large")
        with left_column:
            render_chart_frame_start()
            st.plotly_chart(
                build_validation_figure(validation_summary),
                use_container_width=True,
                key="pipeline_validation_figure",
            )
            render_chart_frame_end()
        with right_column:
            render_subpanel_start()
            st.dataframe(validation_summary, use_container_width=True, hide_index=True, height=320)
            render_subpanel_end()


def render_pipeline_hyperparameter_tuning(engineered_dataframe: pd.DataFrame) -> None:
    """Render simple hyperparameter tuning for cluster count."""
    with st.expander("Hyperparameter Tuning", expanded=True):
        render_section_header(
            "Hyperparameter Tuning",
            "Evaluate how silhouette quality changes across candidate cluster counts while keeping the current financial feature set intact.",
        )
        controls_left, controls_right = st.columns(2, gap="large")
        with controls_left:
            st.session_state.tuning_cluster_min = st.slider(
                "Minimum clusters",
                min_value=2,
                max_value=8,
                value=int(st.session_state.tuning_cluster_min),
            )
        with controls_right:
            max_allowed = max(st.session_state.tuning_cluster_min, 2)
            st.session_state.tuning_cluster_max = st.slider(
                "Maximum clusters",
                min_value=max_allowed,
                max_value=10,
                value=max(int(st.session_state.tuning_cluster_max), max_allowed),
            )

        tuning_summary = st.session_state.model_metrics.get("tuning_summary", pd.DataFrame())
        chart_column, table_column = st.columns([1.35, 0.95], gap="large")
        with chart_column:
            render_chart_frame_start()
            st.plotly_chart(
                build_tuning_figure(tuning_summary),
                use_container_width=True,
                key="pipeline_tuning_figure",
            )
            render_chart_frame_end()
        with table_column:
            render_subpanel_start()
            st.dataframe(tuning_summary, use_container_width=True, hide_index=True, height=320)
            render_subpanel_end()


def render_pipeline_model(engineered_dataframe: pd.DataFrame) -> None:
    """Render model selection, training, and evaluation."""
    with st.expander("Model Selection", expanded=True):
        render_section_header(
            "Model Selection",
            "Configure KMeans using the selected factor set and set the number of market regimes to detect.",
        )
        selected_columns = st.session_state.selected_feature_columns
        if not selected_columns:
            st.info("No features are available after selection.")
            return

        controls_column, notes_column = st.columns([1.1, 0.9], gap="large")
        with controls_column:
            st.session_state.model_features = st.multiselect(
                "KMeans features",
                options=selected_columns,
                default=[
                    column for column in st.session_state.model_features if column in selected_columns
                ]
                or selected_columns[: min(3, len(selected_columns))],
            )
            max_clusters = max(2, min(10, len(engineered_dataframe)))
            st.session_state.cluster_count = 3 if max_clusters >= 3 else max_clusters
            st.metric("Number of clusters", f"{st.session_state.cluster_count}")
        with notes_column:
            st.info("Clustering is trained on complete numeric rows using StandardScaler and fixed three-regime setup when data allows.")

    with st.expander("Training & Evaluation", expanded=True):
        render_section_header(
            "Training & Evaluation",
            "Review the labeled output frame and clustering diagnostics in one consolidated evaluation block.",
        )
        modeled_dataframe = st.session_state.modeled_dataframe
        metrics = st.session_state.model_metrics
        inertia = metrics.get("inertia")
        silhouette = metrics.get("silhouette_score")
        davies_bouldin = metrics.get("davies_bouldin_score")
        training_rows = int(metrics.get("training_row_count", 0))
        testing_rows = int(metrics.get("testing_row_count", 0))
        labeled_rows = int(metrics.get("labeled_row_count", 0))

        render_metric_row(
            [
                ("Training Rows", f"{training_rows:,}", "Rows used to fit KMeans"),
                ("Evaluation Rows", f"{testing_rows:,}", "Valid later-period rows scored after training"),
                ("Inertia", f"{inertia:.2f}" if inertia is not None else "N/A", "Within-cluster compactness"),
                (
                    "Silhouette",
                    f"{silhouette:.3f}" if silhouette is not None else "N/A",
                    "Cluster separation quality",
                ),
            ]
        )
        render_spacer()
        content_column, notes_column = st.columns([1.35, 0.9], gap="large")
        with content_column:
            if modeled_dataframe is None or "market_regime" not in modeled_dataframe.columns:
                st.info("Select at least two valid model features to enable clustering.")
            else:
                display_columns = st.session_state.model_metrics.get("model_features", []) + ["market_regime"]
                render_subpanel_start()
                st.dataframe(
                    modeled_dataframe[display_columns].head(12),
                    use_container_width=True,
                    height=320,
                )
                render_subpanel_end()
        with notes_column:
            render_subpanel_start()
            st.write(
                "Because this is an unsupervised clustering problem, classification metrics such as accuracy, precision, recall, and F1 are not the primary evaluation metrics."
            )
            st.write(f"Labeled rows available after feature completeness filtering: {labeled_rows:,}.")
            if davies_bouldin is not None:
                st.write(f"Davies-Bouldin Score: {davies_bouldin:.3f}")
            if silhouette is None:
                st.info("Silhouette score is unavailable when the clustering sample is too small.")
            elif silhouette >= 0.5:
                st.success("The clustering structure is relatively well separated.")
            elif silhouette >= 0.25:
                st.warning("The clustering structure is moderate and may benefit from feature tuning.")
            else:
                st.warning("Cluster separation is weak. Review the selected features and cluster count.")
            render_subpanel_end()
        render_spacer()
        render_subpanel_start()
        st.write(
            "Too few clusters can underfit by merging distinct market states, while too many clusters can overfit short-lived noise patterns. Use silhouette score, Davies-Bouldin score, and regime stability across TimeSeriesSplit folds to judge the right balance."
        )
        render_subpanel_end()


def render_ml_pipeline_page() -> None:
    """Render the ML pipeline page."""
    render_page_header(
        "Model",
        "Model Console",
        "Keep the academic workflow intact while presenting preprocessing, validation, and tuning as a cleaner model operations console.",
    )

    raw_dataframe = st.session_state.raw_dataframe
    if raw_dataframe is None:
        st.info("Upload a dataset in Dataset Overview to begin the pipeline.")
        return

    compute_pipeline_outputs()
    engineered_dataframe = st.session_state.engineered_dataframe
    modeled_dataframe = st.session_state.modeled_dataframe
    regime_count = 0
    if modeled_dataframe is not None and "market_regime" in modeled_dataframe.columns:
        regime_count = int(modeled_dataframe["market_regime"].dropna().nunique())

    render_metric_row(
        [
            ("Processed Rows", f"{len(st.session_state.cleaned_dataframe):,}", "Rows after cleaning stage"),
            ("Engineered Columns", f"{len(engineered_dataframe.columns):,}", "Feature space after transformation"),
            ("Selected Features", f"{len(st.session_state.selected_feature_columns):,}", "Features retained for clustering"),
            ("Detected Regimes", f"{regime_count:,}", "Non-null market regime labels"),
        ]
    )
    render_spacer("xl")
    render_pipeline_summary()
    render_spacer()
    render_pipeline_problem_setup()
    render_spacer()
    render_panel_start()
    render_pipeline_eda(engineered_dataframe)
    render_pipeline_cleaning(raw_dataframe)
    compute_pipeline_outputs()
    render_pipeline_feature_engineering(st.session_state.cleaned_dataframe)
    compute_pipeline_outputs()
    render_pipeline_feature_selection(st.session_state.engineered_dataframe)
    compute_pipeline_outputs()
    render_pipeline_data_split(st.session_state.engineered_dataframe)
    compute_pipeline_outputs()
    render_pipeline_model(st.session_state.engineered_dataframe)
    compute_pipeline_outputs()
    render_pipeline_validation(st.session_state.engineered_dataframe)
    compute_pipeline_outputs()
    render_pipeline_hyperparameter_tuning(st.session_state.engineered_dataframe)
    render_panel_end()


def get_processed_dataframe() -> pd.DataFrame | None:
    """Return the best available processed dataset."""
    if st.session_state.raw_dataframe is None:
        return None
    compute_pipeline_outputs()
    return st.session_state.modeled_dataframe


def render_market_regime_analysis_page() -> None:
    """Render market regime analytics."""
    render_page_header(
        "Quantitative Tools",
        "Regime Monitor",
        "Track price structure, state persistence, and signal intensity through a chart-first market monitoring view.",
    )

    processed_dataframe = get_processed_dataframe()
    if processed_dataframe is None:
        st.info("Upload a dataset and configure the ML Pipeline first.")
        return

    numerical_columns = get_numerical_columns(processed_dataframe)
    silhouette = st.session_state.model_metrics.get("silhouette_score")
    if "market_regime" not in processed_dataframe.columns or not processed_dataframe["market_regime"].notna().any():
        render_panel_start()
        st.info("Train KMeans in the ML Pipeline page to visualize market regimes.")
        render_panel_end()
        return

    default_price = (
        st.session_state.price_column if st.session_state.price_column in numerical_columns else ("Close" if "Close" in numerical_columns else numerical_columns[0])
    )
    inferred_time = st.session_state.model_metrics.get("time_column") or st.session_state.date_column or infer_time_column(processed_dataframe)
    time_column_options = get_time_column_options(processed_dataframe)
    _, _, default_mapping_items = compute_regime_semantics_cached(
        processed_dataframe,
        default_price,
        inferred_time,
    )
    default_regime_map = dict(default_mapping_items)
    regime_rows = int(st.session_state.model_metrics.get("labeled_row_count", processed_dataframe["market_regime"].notna().sum()))
    regime_count = int(processed_dataframe["market_regime"].dropna().nunique())
    training_rows = int(st.session_state.model_metrics.get("training_row_count", 0))
    regime_mode = int(
        pd.to_numeric(processed_dataframe["market_regime"], errors="coerce")
        .dropna()
        .astype(int)
        .mode()
        .iloc[0]
    )
    current_regime = int(
        pd.to_numeric(processed_dataframe["market_regime"], errors="coerce")
        .dropna()
        .astype(int)
        .iloc[-1]
    )
    mode_semantic = default_regime_map.get(regime_mode, f"Regime {regime_mode}")
    current_semantic = default_regime_map.get(current_regime, f"Regime {current_regime}")
    render_metric_row(
        [
            ("Regimes", f"{regime_count:,}", "Distinct detected market states"),
            ("Silhouette Score", f"{silhouette:.3f}" if silhouette is not None else "N/A", "Cluster separation quality"),
            ("Most Common Regime", f"{mode_semantic} (R{regime_mode})", "Highest-frequency labeled market state"),
            ("Current Regime", f"{current_semantic} (R{current_regime})", "Most recent detected market state"),
            ("Market Coverage", f"{training_rows:,} / {regime_rows:,}", "Training sample versus all labeled market rows"),
        ]
    )
    render_spacer("xl")

    render_control_strip_start()
    render_section_header(
        "Monitor Controls",
        "Tune the active market series and view window without taking attention away from the chart.",
    )
    controls_column, config_column = st.columns(2, gap="large")
    with controls_column:
        price_column = st.selectbox(
            "Price column",
            options=numerical_columns,
            index=numerical_columns.index(default_price),
        )
    with config_column:
        time_options = ["Use Index"] + time_column_options
        selected_time = st.selectbox(
            "Date column",
            options=time_options,
            index=(time_options.index(inferred_time) if inferred_time else 0),
        )
    view_column, mode_column, window_column = st.columns(3, gap="large")
    with view_column:
        chart_view = st.selectbox(
            "Chart view",
            options=["Recent Regimes", "Full History"],
            index=0,
        )
    with mode_column:
        display_mode = st.selectbox(
            "Display mode",
            options=["Raw Price", "Log Scale", "Rebased"],
            index=0 if chart_view == "Recent Regimes" else 1,
        )
    with window_column:
        recent_periods = st.slider(
            "Recent observations",
            min_value=150,
            max_value=1500,
            value=500,
            step=50,
            disabled=chart_view != "Recent Regimes",
        )
    render_control_strip_end()

    selected_time_column = None if selected_time == "Use Index" else selected_time
    semantic_dataframe, _, mapping_items = compute_regime_semantics_cached(
        processed_dataframe,
        price_column,
        selected_time_column,
    )
    regime_state_map = dict(mapping_items)
    numeric_regime_series = pd.to_numeric(semantic_dataframe["market_regime"], errors="coerce").dropna().astype(int)
    semantic_label_series = semantic_dataframe["regime_type"].dropna().astype(str) if "regime_type" in semantic_dataframe.columns else pd.Series(dtype=str)
    all_detected_regimes = sorted(numeric_regime_series.unique().tolist())
    numeric_value_counts = (
        numeric_regime_series.value_counts().sort_index().rename_axis("market_regime").reset_index(name="Count")
        if not numeric_regime_series.empty
        else pd.DataFrame(columns=["market_regime", "Count"])
    )
    semantic_value_counts = (
        semantic_label_series.value_counts().rename_axis("regime_type").reset_index(name="Count")
        if not semantic_label_series.empty
        else pd.DataFrame(columns=["regime_type", "Count"])
    )
    regime_debug_summary = compute_regime_statistics(
        semantic_dataframe,
        regime_col="market_regime",
        price_column=price_column,
        time_column=selected_time_column,
    )
    if not regime_debug_summary.empty:
        regime_debug_summary["Regime Type"] = regime_debug_summary["Regime"].map(
            lambda value: regime_state_map.get(int(value), "Neutral")
        )
        regime_debug_summary["Regime"] = regime_debug_summary["Regime"].astype(int)
        regime_debug_summary = regime_debug_summary.rename(
            columns={
                "Count": "Row Count",
                "Mean Return": "Mean Return",
                "Volatility": "Volatility",
            }
        )[
            ["Regime", "Regime Type", "Row Count", "Mean Return", "Volatility"]
        ]

    plot_data = prepare_market_regime_plot_data(
        semantic_dataframe,
        price_column,
        selected_time_column,
    )
    if plot_data.errors:
        for error in plot_data.errors:
            st.warning(error)

    display_frame = (
        slice_recent_period(plot_data.full_series, recent_periods)
        if chart_view == "Recent Regimes" and not plot_data.full_series.empty
        else plot_data.full_series
    )
    display_range_value = f"{display_frame[price_column].min():.2f} to {display_frame[price_column].max():.2f}"
    display_range_label = "Visible raw-price span"
    if display_mode == "Rebased":
        rebased_series = (pd.to_numeric(display_frame[price_column], errors="coerce") / float(pd.to_numeric(display_frame[price_column], errors="coerce").dropna().iloc[0])) * 100.0 if not display_frame.empty and not pd.to_numeric(display_frame[price_column], errors="coerce").dropna().empty else pd.Series(dtype=float)
        if not rebased_series.empty:
            display_range_value = f"{rebased_series.min():.2f} to {rebased_series.max():.2f}"
        display_range_label = "Visible rebased span"
    elif display_mode == "Log Scale":
        positive_series = pd.to_numeric(display_frame[price_column], errors="coerce")
        positive_series = positive_series[positive_series > 0]
        if not positive_series.empty:
            display_range_value = f"{positive_series.min():.2f} to {positive_series.max():.2f}"
        display_range_label = "Visible positive-price span"

    render_hero_chart_panel_start()
    render_section_header(
        "Market State Overlay",
        "Primary market view with price action and detected state shifts rendered on one monitoring surface.",
    )
    if plot_data.full_series.empty:
        st.warning("The market regime chart cannot be rendered with the current selections.")
    else:
        render_metric_row(
            [
                (
                    "Visible Rows",
                    f"{len(display_frame):,}",
                    "Rows currently shown in the chart",
                ),
                (
                    "View Mode",
                    chart_view,
                    "Focused recent window or full market history",
                ),
                (
                    "Display",
                    display_mode,
                    "Raw, log-scaled, or rebased market view",
                ),
                (
                    "Price Range",
                    display_range_value,
                    display_range_label,
                ),
            ]
        )
        render_spacer()
        render_chart_frame_start()
        st.plotly_chart(
            build_market_regime_figure(
                semantic_dataframe,
                price_column,
                selected_time_column,
                recent_only=chart_view == "Recent Regimes",
                recent_periods=recent_periods,
                expected_regimes=all_detected_regimes,
                display_mode=display_mode,
            ),
            use_container_width=True,
            key="regime_monitor_main_chart",
        )
        render_chart_frame_end()
        visible_regimes = (
            pd.to_numeric(display_frame.get("market_regime"), errors="coerce").dropna().astype(int).nunique()
            if "market_regime" in display_frame.columns
            else 0
        )
        if chart_view == "Recent Regimes" and len(all_detected_regimes) > max(visible_regimes, 0):
            st.info(
                f"Recent window currently shows {visible_regimes} regime(s), while {len(all_detected_regimes)} regime(s) exist overall. Switch to Full History to inspect all states directly."
            )
        render_spacer()
        render_subpanel_start()
        st.markdown('<div class="inline-panel-title">State Timeline</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="inline-panel-copy">Recent regime occupancy strip for faster reading of state persistence and switching behavior.</div>',
            unsafe_allow_html=True,
        )
        render_chart_frame_start()
        st.plotly_chart(
            build_regime_timeline_figure(
                semantic_dataframe,
                time_column=selected_time_column,
                regime_labels=regime_state_map,
                recent_periods=recent_periods if chart_view == "Recent Regimes" else 260,
            ),
            use_container_width=True,
            key="regime_monitor_timeline",
        )
        render_chart_frame_end()
        render_subpanel_end()
    render_panel_end()

    render_spacer()
    render_panel_start(compact=True)
    render_section_header(
        "State Diagnostics",
        "Compact monitor checks for label coverage, dominance, and feature health behind the visible regime view.",
    )
    debug_metrics = [
        ("Unique Numeric Regimes", f"{len(all_detected_regimes):,}", "Distinct values in market_regime"),
        ("Unique Semantic Labels", f"{semantic_label_series.nunique():,}", "Distinct values in regime_type"),
        ("Mapped Regimes", f"{len(regime_state_map):,}", "Number of numeric regimes mapped to semantic labels"),
    ]
    render_metric_row(debug_metrics)
    if not numeric_value_counts.empty:
        with st.expander("Open detailed state diagnostics", expanded=False):
            left_debug, right_debug = st.columns(2, gap="large")
            with left_debug:
                render_subpanel_start()
                st.caption("market_regime value counts")
                st.dataframe(numeric_value_counts, use_container_width=True, hide_index=True, height=220)
                render_subpanel_end()
            with right_debug:
                render_subpanel_start()
                st.caption("regime_type value counts")
                st.dataframe(semantic_value_counts, use_container_width=True, hide_index=True, height=220)
                render_subpanel_end()
    if len(regime_state_map) and len(set(regime_state_map.values())) < len(regime_state_map):
        st.warning(
            "Multiple numeric clusters currently map to the same semantic label. This can be valid in weak-separation markets, but review feature variance and silhouette for confirmation."
        )
    model_features = st.session_state.model_metrics.get("model_features", [])
    if model_features:
        feature_variance = semantic_dataframe[model_features].apply(pd.to_numeric, errors="coerce").var(ddof=0).fillna(0.0)
        low_variance_features = feature_variance[feature_variance <= 1e-6].index.tolist()
        if low_variance_features:
            st.warning(
                f"Low-variation model features detected: {', '.join(low_variance_features)}. This can collapse clustering into fewer effective regimes."
            )
    if not regime_debug_summary.empty:
        with st.expander("Open numeric state summary", expanded=False):
            render_subpanel_start()
            st.caption("Regime summary table (numeric + semantic)")
            st.dataframe(regime_debug_summary.round(4), use_container_width=True, hide_index=True, height=240)
            render_subpanel_end()
    render_panel_end()

    render_panel_start()
    render_section_header(
        "Market Structure",
        "Read state balance and transition persistence together so the supporting diagnostics feel like one integrated monitor.",
    )
    outlook_table = compute_current_regime_outlook(semantic_dataframe, regime_labels=regime_state_map)
    if not outlook_table.empty:
        current_state_name = regime_state_map.get(int(outlook_table["Current Regime Id"].iloc[0]), f"Regime {int(outlook_table['Current Regime Id'].iloc[0])}")
        stay_probability = float(outlook_table.loc[outlook_table["Regime Id"] == int(outlook_table["Current Regime Id"].iloc[0]), "Probability"].iloc[0]) if (outlook_table["Regime Id"] == int(outlook_table["Current Regime Id"].iloc[0])).any() else 0.0
        alternate_rows = outlook_table.loc[outlook_table["Regime Id"] != int(outlook_table["Current Regime Id"].iloc[0])]
        likely_switch_label = alternate_rows.iloc[0]["Regime"] if not alternate_rows.empty else outlook_table.iloc[0]["Regime"]
        likely_switch_prob = float(alternate_rows.iloc[0]["Probability"]) if not alternate_rows.empty else 0.0
        render_metric_row(
            [
                ("Current State", current_state_name, "Latest semantic state"),
                ("Stay Probability", f"{stay_probability:.1%}", "Probability of remaining in the current state next step"),
                ("Likely Switch", str(likely_switch_label), "Highest-probability alternate transition"),
                ("Switch Probability", f"{likely_switch_prob:.1%}", "Most likely alternate next-state probability"),
            ]
        )
        render_spacer()
    support_left, support_right = st.columns(2, gap="large")
    with support_left:
        render_subpanel_start()
        st.markdown('<div class="inline-panel-title">State Balance</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="inline-panel-copy">Observation balance across detected states for a quick read on concentration and market coverage.</div>',
            unsafe_allow_html=True,
        )
        render_chart_frame_start()
        st.plotly_chart(
            build_cluster_distribution_figure(semantic_dataframe, regime_labels=regime_state_map),
            use_container_width=True,
            key="regime_monitor_cluster_distribution",
        )
        render_chart_frame_end()
        render_subpanel_end()
    with support_right:
        render_subpanel_start()
        st.markdown('<div class="inline-panel-title">State Transition</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="inline-panel-copy">Chronological transition probabilities arranged for faster reading of persistence and switching behavior.</div>',
            unsafe_allow_html=True,
        )
        render_chart_frame_start()
        st.plotly_chart(
            build_regime_transition_figure(semantic_dataframe, regime_labels=regime_state_map),
            use_container_width=True,
            key="regime_monitor_transition_matrix",
        )
        render_chart_frame_end()
        render_subpanel_end()
    render_panel_end()

    if not outlook_table.empty:
        render_spacer()
        render_panel_start(compact=True)
        render_section_header(
            "Transition Outlook",
            "Empirical next-step transition probabilities from the current state, based on observed state changes in the time series.",
        )
        render_chart_frame_start()
        st.plotly_chart(
            build_current_regime_outlook_figure(semantic_dataframe, regime_labels=regime_state_map),
            use_container_width=True,
            key="regime_monitor_current_outlook",
        )
        render_chart_frame_end()
        render_panel_end()

        render_spacer()
        render_panel_start(compact=True)
        render_section_header(
            "State Forecast",
            "A Markov-style projection of likely regime occupancy several steps ahead, based on the observed transition structure.",
        )
        forecast_horizon = st.slider(
            "Forecast horizon",
            min_value=2,
            max_value=10,
            value=3,
            step=1,
            key="markov_forecast_horizon",
        )
        forecast_table = compute_markov_regime_forecast(
            semantic_dataframe,
            steps_ahead=int(forecast_horizon),
            regime_labels=regime_state_map,
        )
        if not forecast_table.empty:
            current_state_id = int(forecast_table["Current Regime Id"].iloc[0])
            current_state_name = regime_state_map.get(current_state_id, f"Regime {current_state_id}")
            dominant_forecast = forecast_table.iloc[0]
            persistence_prob = float(
                forecast_table.loc[forecast_table["Regime Id"] == current_state_id, "Probability"].iloc[0]
            ) if (forecast_table["Regime Id"] == current_state_id).any() else 0.0
            render_metric_row(
                [
                    ("Horizon", f"{forecast_horizon} step", "Forecast look-ahead window"),
                    ("Current State", current_state_name, "Most recent detected state"),
                    ("Likely Future State", str(dominant_forecast["Regime"]), "Highest-probability forecast state"),
                    ("Persistence", f"{persistence_prob:.1%}", "Probability the current state still dominates at the selected horizon"),
                ]
            )
            render_spacer()
            render_chart_frame_start()
            st.plotly_chart(
                build_markov_regime_forecast_figure(
                    semantic_dataframe,
                    steps_ahead=int(forecast_horizon),
                    regime_labels=regime_state_map,
                ),
                use_container_width=True,
                key="regime_monitor_markov_forecast",
            )
            render_chart_frame_end()
            with st.expander("Open forecast probabilities", expanded=False):
                st.dataframe(
                    forecast_table[["Regime", "Probability"]].assign(
                        Probability=lambda frame: frame["Probability"].round(4)
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            st.info("Not enough transition history is available to build a multi-step regime forecast.")
        render_panel_end()

    render_spacer()
    render_panel_start()
    render_section_header(
        "Signal Profile",
        "Use state-level return, volatility, and feature intensity to explain what each detected market condition represents.",
    )
    summary_left, summary_right = st.columns([0.95, 1.05], gap="large")
    regime_summary = build_regime_summary_table(
        semantic_dataframe,
        price_column,
        selected_time_column,
    )
    with summary_left:
        render_subpanel_start()
        st.markdown('<div class="inline-panel-title">State Summary</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="inline-panel-copy">Average return, volatility, and observed duration translated into readable market-state diagnostics.</div>',
            unsafe_allow_html=True,
        )
        if not regime_summary.empty:
            summary_metrics = []
            for _, row in regime_summary.iterrows():
                summary_metrics.append(
                    (
                        str(row["Regime"]),
                        str(row["Regime Type"]),
                        f"Ret {row['Average Return']:.2%} | Vol {row['Volatility']:.2%} | Dur {row['Average Duration']}",
                    )
                )
            render_metric_row(summary_metrics[: min(3, len(summary_metrics))])
            render_spacer()
        with st.expander("Open full state summary table", expanded=False):
            st.dataframe(regime_summary, use_container_width=True, hide_index=True, height=300)
        render_subpanel_end()
    with summary_right:
        render_subpanel_start()
        st.markdown('<div class="inline-panel-title">Signal Profile</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="inline-panel-copy">A centroid-style view of which modeling features are elevated or muted within each regime.</div>',
            unsafe_allow_html=True,
        )
        feature_profile_columns = get_feature_profile_columns(semantic_dataframe)
        if feature_profile_columns:
            st.caption(f"Profile features: {', '.join(feature_profile_columns)}")
            render_chart_frame_start()
            st.plotly_chart(
                build_regime_feature_profile_figure(semantic_dataframe, feature_profile_columns),
                use_container_width=True,
                key="regime_monitor_feature_profile",
            )
            render_chart_frame_end()
        else:
            st.warning("No suitable non-redundant features are available for the regime feature profile.")
        render_subpanel_end()
    render_panel_end()

    render_spacer()
    render_panel_start()
    render_section_header(
        "Desk Signals",
        "Translate the state model into a compact desk-style view of current condition, state confidence, and interpretable diagnostics.",
    )
    insights_left, insights_right = st.columns([1.15, 0.85], gap="large")
    regime_insights = build_regime_insights_table(
        semantic_dataframe,
        price_column,
        selected_time_column,
    )
    with insights_left:
        render_subpanel_start()
        st.markdown('<div class="inline-panel-title">State Insight Grid</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="inline-panel-copy">Average return, volatility, data density, and price-change behavior with semantic state interpretation.</div>',
            unsafe_allow_html=True,
        )
        if not regime_insights.empty:
            insight_metrics = []
            for _, row in regime_insights.iterrows():
                insight_metrics.append(
                    (
                        str(row["Regime"]),
                        str(row["Regime Type"]),
                        f"Pts {row['Data Points']} | Ret {row['Average Return']:.2%} | Vol {row['Volatility']:.2%}",
                    )
                )
            render_metric_row(insight_metrics[: min(3, len(insight_metrics))])
            render_spacer()
        with st.expander("Open detailed state insight table", expanded=False):
            st.dataframe(regime_insights, use_container_width=True, hide_index=True, height=300)
        render_subpanel_end()
    with insights_right:
        render_subpanel_start()
        st.markdown('<div class="inline-panel-title">Desk Read</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="inline-panel-copy">A centroid-similarity read on the latest feature vector, plus the nearest alternate state. This is a state estimate, not a next-state forecast.</div>',
            unsafe_allow_html=True,
        )
        model_features = st.session_state.model_metrics.get("model_features", [])
        if len(model_features) >= 2:
            pred_current, pred_next, pred_confidence = compute_regime_prediction_cached(
                processed_dataframe,
                tuple(model_features),
                int(st.session_state.cluster_count),
                st.session_state.model_metrics.get("time_column"),
                float(st.session_state.test_split_ratio),
            )
            pred_col_a, pred_col_b = st.columns(2, gap="medium")
            with pred_col_a:
                semantic_current = regime_state_map.get(pred_current, "N/A") if pred_current is not None else "N/A"
                st.metric("Current Regime", f"{semantic_current} (R{pred_current})" if pred_current is not None else "N/A")
            with pred_col_b:
                semantic_next = regime_state_map.get(pred_next, "N/A") if pred_next is not None else "N/A"
                st.metric("Nearest Alternate", f"{semantic_next} (R{pred_next})" if pred_next is not None else "N/A")
            st.metric(
                "Confidence",
                f"{pred_confidence * 100:.1f}%" if pred_confidence is not None else "N/A",
            )
            confidence_history = compute_regime_confidence_history_cached(
                processed_dataframe,
                tuple(model_features),
                int(st.session_state.cluster_count),
                st.session_state.model_metrics.get("time_column"),
                float(st.session_state.test_split_ratio),
            )
            if not confidence_history.empty:
                render_chart_frame_start()
                st.plotly_chart(
                    build_confidence_history_figure(confidence_history.tail(220)),
                    use_container_width=True,
                    key="regime_monitor_confidence_history",
                )
                render_chart_frame_end()
        else:
            st.info("Select at least two modeling features in ML Pipeline to enable the regime-estimate diagnostics.")
        render_subpanel_end()
    render_panel_end()

    render_spacer()
    render_panel_start(compact=True)
    render_section_header(
        "Scenario Probe",
        "Input hypothetical factor values and map the scenario to the closest detected market state under the current setup.",
    )
    model_features = st.session_state.model_metrics.get("model_features", [])
    feature_aliases = {
        "return": next((value for value in model_features if "return" in value.lower()), None),
        "volatility": next((value for value in model_features if "volatility" in value.lower()), None),
        "rolling_mean": next((value for value in model_features if "rolling_mean" in value.lower() or "rolling mean" in value.lower()), None),
    }
    if len(model_features) >= 2 and all(feature_aliases.values()):
        scenario_left, scenario_right, scenario_meta = st.columns([1.0, 1.0, 0.9], gap="large")
        with scenario_left:
            st.session_state.what_if_return = st.number_input(
                "Hypothetical Return",
                value=float(st.session_state.what_if_return),
                step=0.001,
                format="%.4f",
            )
        with scenario_right:
            st.session_state.what_if_volatility = st.number_input(
                "Hypothetical Volatility",
                value=float(st.session_state.what_if_volatility),
                step=0.001,
                format="%.4f",
                min_value=0.0,
            )
        with scenario_meta:
            st.session_state.what_if_rolling_mean = st.number_input(
                "Hypothetical Rolling Mean",
                value=float(st.session_state.what_if_rolling_mean),
                step=0.1,
                format="%.4f",
            )
        scenario_inputs = {
            feature_aliases["return"]: float(st.session_state.what_if_return),
            feature_aliases["volatility"]: float(st.session_state.what_if_volatility),
            feature_aliases["rolling_mean"]: float(st.session_state.what_if_rolling_mean),
        }
        scenario_regime, scenario_confidence = compute_what_if_prediction_cached(
            semantic_dataframe,
            tuple(model_features),
            int(st.session_state.cluster_count),
            st.session_state.model_metrics.get("time_column"),
            float(st.session_state.test_split_ratio),
            tuple(sorted(scenario_inputs.items())),
        )
        res_a, res_b = st.columns(2, gap="large")
        with res_a:
            scenario_semantic = regime_state_map.get(scenario_regime, "N/A") if scenario_regime is not None else "N/A"
            st.metric(
                "Scenario Regime",
                f"{scenario_semantic} (R{scenario_regime})" if scenario_regime is not None else "N/A",
            )
        with res_b:
            st.metric("Scenario Confidence", f"{scenario_confidence * 100:.1f}%" if scenario_confidence is not None else "N/A")
    else:
        st.info("What-if simulation needs model features containing return, volatility, and rolling-mean style factors.")
    render_panel_end()

    render_panel_start(compact=True)
    render_section_header(
        "Monitor Notes",
        "Use transition persistence, state-level return and volatility, and the signal profile together to explain why the monitor is economically meaningful.",
    )
    st.info(
        "The system first detects latent clusters using KMeans and then interprets them as Bull, Bear, or Sideways using return and volatility characteristics."
    )
    st.info("Stable transition diagonals usually indicate persistent states, while sharp feature-profile differences help interpret what each regime represents.")
    render_panel_end()


def render_correlation_matrix_page() -> None:
    """Render quantitative correlation analytics."""
    render_page_header(
        "Quantitative Tools",
        "Market Structure",
        "Track feature co-movement through a large-format structure map tuned for fast visual reading.",
    )

    processed_dataframe = get_processed_dataframe()
    if processed_dataframe is None:
        st.info("Upload a dataset and configure the ML Pipeline first.")
        return

    numerical_columns = get_numerical_columns(processed_dataframe)
    if len(numerical_columns) < 2:
        render_panel_start()
        st.warning("At least two numerical columns are required to render the correlation matrix.")
        render_panel_end()
        return

    render_control_strip_start()
    render_section_header(
        "Structure Controls",
        "Choose the processed fields to include while keeping the structure map large and legible.",
    )
    selected_columns = st.multiselect(
        "Selected features",
        options=numerical_columns,
        default=st.session_state.final_model_feature_set[: min(8, len(st.session_state.final_model_feature_set))]
        or st.session_state.selected_feature_columns[: min(8, len(st.session_state.selected_feature_columns))]
        or numerical_columns[: min(8, len(numerical_columns))],
    )
    render_control_strip_end()

    if len(selected_columns) >= 2:
        render_metric_row(
            [
                ("Features", f"{len(selected_columns):,}", "Fields active in the structure map"),
                ("Strongest Pair", "Live", "Use the map to inspect highest positive or negative co-movement"),
            ]
        )
        render_spacer()

    render_hero_chart_panel_start()
    render_section_header(
        "Structure Heatmap",
        "Large-format co-movement view designed to keep the matrix as the primary visual surface.",
    )
    if len(selected_columns) >= 2:
        render_chart_frame_start()
        st.plotly_chart(
            build_quant_correlation_figure(processed_dataframe, selected_columns),
            use_container_width=True,
            key="correlation_matrix_heatmap",
        )
        render_chart_frame_end()
    else:
        st.info("Select at least two features to display the correlation matrix.")
    render_panel_end()

    render_panel_start(compact=True)
    render_section_header(
        "Monitor Note",
        "Keep this as a lightweight support block so the heatmap remains the dominant visual on the page.",
    )
    st.info("Use the heatmap to highlight strongly positive or negative feature relationships before discussing feature selection or regime behavior.")
    render_panel_end()


def render_volatility_analysis_page() -> None:
    """Render rolling volatility analytics."""
    render_page_header(
        "Quantitative Tools",
        "Volatility Monitor",
        "Inspect realized volatility dynamics through a cleaner time-series view built for market surveillance.",
    )

    processed_dataframe = get_processed_dataframe()
    if processed_dataframe is None:
        st.info("Upload a dataset and configure the ML Pipeline first.")
        return

    numerical_columns = get_numerical_columns(processed_dataframe)
    inferred_time = st.session_state.model_metrics.get("time_column") or st.session_state.date_column or infer_time_column(processed_dataframe)
    default_price = st.session_state.price_column if st.session_state.price_column in numerical_columns else ("Close" if "Close" in numerical_columns else numerical_columns[0])
    if not numerical_columns:
        render_panel_start()
        st.warning("No numerical columns are available for volatility analysis.")
        render_panel_end()
        return

    render_control_strip_start()
    render_section_header(
        "Volatility Controls",
        "Tune the source series and lookback window used in the rolling volatility monitor.",
    )
    price_column, window_column = st.columns(2, gap="large")
    with price_column:
        selected_price = st.selectbox(
            "Volatility price column",
            options=numerical_columns,
            index=numerical_columns.index(default_price),
        )
    with window_column:
        window = st.slider("Volatility window", min_value=2, max_value=60, value=20)
    render_control_strip_end()

    returns_series = pd.to_numeric(processed_dataframe[selected_price], errors="coerce").pct_change()
    rolling_vol = returns_series.rolling(window=window, min_periods=2).std()
    valid_rolling_vol = rolling_vol.dropna()
    latest_vol = valid_rolling_vol.iloc[-1] if not valid_rolling_vol.empty else None
    avg_vol = valid_rolling_vol.mean() if not valid_rolling_vol.empty else None
    render_metric_row(
        [
            ("Window", f"{window}", "Active lookback"),
            ("Latest Vol", f"{latest_vol:.2%}" if latest_vol is not None else "N/A", "Most recent rolling observation"),
            ("Average Vol", f"{avg_vol:.2%}" if avg_vol is not None else "N/A", "Mean rolling volatility across the visible series"),
        ]
    )
    render_spacer()

    render_hero_chart_panel_start()
    render_section_header(
        "Rolling Volatility",
        "Primary market monitor showing realized volatility on the currently processed series.",
    )
    render_chart_frame_start()
    st.plotly_chart(
        build_rolling_volatility_figure(processed_dataframe, selected_price, window, inferred_time),
        use_container_width=True,
        key="volatility_monitor_chart",
    )
    render_chart_frame_end()
    render_panel_end()

    render_panel_start(compact=True)
    render_section_header(
        "Desk Read",
        "A compact support block that adds context without competing with the primary volatility monitor.",
    )
    st.info("Watch for sustained volatility expansion or compression zones when connecting this view back to market regimes.")
    render_panel_end()


def render_returns_analysis_page() -> None:
    """Render return-distribution and cumulative-return analytics."""
    render_page_header(
        "Quantitative Tools",
        "Return Monitor",
        "Review cumulative performance and return dispersion using the same processed market dataset.",
    )

    processed_dataframe = get_processed_dataframe()
    if processed_dataframe is None:
        st.info("Upload a dataset and configure the ML Pipeline first.")
        return

    numerical_columns = get_numerical_columns(processed_dataframe)
    inferred_time = st.session_state.model_metrics.get("time_column") or st.session_state.date_column or infer_time_column(processed_dataframe)
    default_price = st.session_state.price_column if st.session_state.price_column in numerical_columns else ("Close" if "Close" in numerical_columns else numerical_columns[0])
    returns_candidates = [column for column in processed_dataframe.columns if column.endswith("_returns")]
    if not numerical_columns:
        render_panel_start()
        st.warning("No numerical columns are available for returns analysis.")
        render_panel_end()
        return

    render_control_strip_start()
    render_section_header(
        "Return Controls",
        "Select an engineered returns field, or derive one dynamically from a chosen price series.",
    )
    if returns_candidates:
        returns_dataframe = processed_dataframe
        returns_column = st.selectbox("Returns column", options=returns_candidates)
    else:
        source_column = st.selectbox(
            "Returns source column",
            options=numerical_columns,
            index=numerical_columns.index(default_price),
        )
        returns_dataframe = processed_dataframe.copy()
        returns_dataframe["dashboard_returns"] = returns_dataframe[source_column].pct_change()
        returns_column = "dashboard_returns"
    render_control_strip_end()

    valid_returns = pd.to_numeric(returns_dataframe[returns_column], errors="coerce").dropna()
    cumulative_tail = (1 + valid_returns).cumprod() - 1 if not valid_returns.empty else pd.Series(dtype=float)
    render_metric_row(
        [
            ("Observations", f"{len(valid_returns):,}", "Return points in the active series"),
            ("Mean Return", f"{valid_returns.mean():.2%}" if not valid_returns.empty else "N/A", "Average period return"),
            ("Volatility", f"{valid_returns.std(ddof=0):.2%}" if len(valid_returns) > 1 else "N/A", "Return dispersion"),
            ("Cumulative", f"{cumulative_tail.iloc[-1]:.2%}" if not cumulative_tail.empty else "N/A", "Compounded return of the active series"),
        ]
    )
    render_spacer()

    left_column, right_column = st.columns(2, gap="large")
    with left_column:
        render_hero_chart_panel_start()
        render_section_header(
            "Cumulative Returns",
            "Compounded performance path over time for the active return stream.",
        )
        render_chart_frame_start()
        st.plotly_chart(
            build_cumulative_returns_figure(returns_dataframe, returns_column, inferred_time),
            use_container_width=True,
            key="returns_monitor_cumulative",
        )
        render_chart_frame_end()
        render_panel_end()
    with right_column:
        render_panel_start()
        render_section_header(
            "Returns Distribution",
            "Frequency profile of period-to-period returns to highlight skew and dispersion.",
        )
        render_chart_frame_start()
        st.plotly_chart(
            build_returns_histogram(returns_dataframe, returns_column),
            use_container_width=True,
            key="returns_monitor_histogram",
        )
        render_chart_frame_end()
        render_panel_end()

    render_panel_start(compact=True)
    render_section_header(
        "Desk Read",
        "Use cumulative returns for directional story and the histogram to discuss dispersion, skew, and tail behavior.",
    )
    st.info("Use the left chart for market narrative and the distribution panel to explain how noisy or asymmetric the return stream is.")
    render_panel_end()


def render_trading_strategy_page() -> None:
    """Render regime-driven trading strategy simulation."""
    render_page_header(
        "Quantitative Tools",
        "Strategy Desk",
        "Test regime-aware positioning against a passive benchmark using the current state model and processed return stream.",
    )

    processed_dataframe = get_processed_dataframe()
    if processed_dataframe is None:
        st.info("Upload a dataset and configure the ML Pipeline first.")
        return
    if "market_regime" not in processed_dataframe.columns or not processed_dataframe["market_regime"].notna().any():
        st.warning("Regime labels are missing. Train the clustering model in ML Pipeline before running strategy simulation.")
        return

    numerical_columns = get_numerical_columns(processed_dataframe)
    if not numerical_columns:
        st.warning("No numerical columns are available for strategy simulation.")
        return

    inferred_time = st.session_state.model_metrics.get("time_column") or st.session_state.date_column or infer_time_column(processed_dataframe)
    default_price = st.session_state.price_column if st.session_state.price_column in numerical_columns else numerical_columns[0]
    returns_candidates = [column for column in processed_dataframe.columns if column.endswith("_returns")]
    render_control_strip_start()
    render_section_header(
        "Strategy Controls",
        "Configure regime exposure, return source, and transaction cost for the trading simulation.",
    )
    source_left, source_right = st.columns(2, gap="large")
    with source_left:
        return_source = st.selectbox(
            "Return source",
            options=["Compute from price column"] + returns_candidates,
            index=0,
        )
    with source_right:
        selected_price = st.selectbox(
            "Price column",
            options=numerical_columns,
            index=numerical_columns.index(default_price),
            disabled=return_source != "Compute from price column",
        )
    exp_a, exp_b, exp_c = st.columns(3, gap="large")
    with exp_a:
        bull_exposure = st.slider("Bull exposure", min_value=0.0, max_value=1.0, value=1.0, step=0.05)
    with exp_b:
        bear_exposure = st.slider("Bear exposure", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
    with exp_c:
        sideways_exposure = st.slider("Sideways exposure", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
    transaction_cost_bps = st.number_input("Transaction cost (bps)", min_value=0.0, max_value=100.0, value=0.0, step=1.0)
    render_control_strip_end()

    semantic_dataframe, _, _ = compute_regime_semantics_cached(processed_dataframe, selected_price, inferred_time)
    if "regime_type" not in semantic_dataframe.columns or semantic_dataframe["regime_type"].dropna().empty:
        st.warning("Semantic regime labels are unavailable. Check regime interpretation in Market Regime Analysis.")
        return

    returns_column = None if return_source == "Compute from price column" else return_source
    strategy_input = compute_daily_returns(
        semantic_dataframe,
        price_column=selected_price,
        returns_column=returns_column,
        time_column=inferred_time,
    )
    if strategy_input.empty:
        st.warning("Unable to compute market returns for the selected inputs.")
        return

    strategy_frame = compute_strategy_returns(
        strategy_input,
        bull_exposure=bull_exposure,
        bear_exposure=bear_exposure,
        sideways_exposure=sideways_exposure,
        transaction_cost=float(transaction_cost_bps) / 10000.0,
    )
    if strategy_frame.empty:
        st.warning("Strategy simulation failed because returns or regime labels are incomplete.")
        return

    strategy_frame["Strategy Cumulative"] = compute_cumulative_returns(strategy_frame["Strategy Return"])
    strategy_frame["Buy and Hold Cumulative"] = compute_cumulative_returns(strategy_frame["Market Return"])
    strategy_drawdown = compute_drawdown(strategy_frame["Strategy Cumulative"])
    strategy_volatility = float(pd.to_numeric(strategy_frame["Strategy Return"], errors="coerce").std(ddof=0))
    strategy_sharpe = compute_sharpe_ratio(strategy_frame["Strategy Return"])
    strategy_sortino = compute_sortino_ratio(strategy_frame["Strategy Return"])
    strategy_cagr = compute_cagr(strategy_frame["Strategy Return"])
    hit_rate = compute_hit_rate(strategy_frame["Strategy Return"])
    total_strategy_return = float(strategy_frame["Strategy Cumulative"].iloc[-1]) if not strategy_frame.empty else 0.0
    buy_hold_return = float(strategy_frame["Buy and Hold Cumulative"].iloc[-1]) if not strategy_frame.empty else 0.0
    excess_return = total_strategy_return - buy_hold_return
    max_drawdown = float(strategy_drawdown.min()) if not strategy_drawdown.empty else 0.0

    render_metric_row(
        [
            ("Total Strategy Return", f"{total_strategy_return:.2%}", "Compounded regime strategy performance"),
            ("Buy & Hold Return", f"{buy_hold_return:.2%}", "Compounded passive benchmark return"),
            ("Excess Return", f"{excess_return:.2%}", "Strategy minus benchmark return"),
            ("Max Drawdown", f"{max_drawdown:.2%}", "Worst peak-to-trough strategy drawdown"),
            ("Volatility", f"{strategy_volatility:.2%}", "Strategy return dispersion"),
            ("Sharpe Ratio", f"{strategy_sharpe:.2f}" if strategy_sharpe is not None else "N/A", "Simple annualized risk-adjusted return"),
            ("Sortino", f"{strategy_sortino:.2f}" if strategy_sortino is not None else "N/A", "Downside-sensitive risk-adjusted return"),
            ("CAGR", f"{strategy_cagr:.2%}" if strategy_cagr is not None else "N/A", "Annualized compounded growth estimate"),
            ("Hit Rate", f"{hit_rate:.1%}" if hit_rate is not None else "N/A", "Share of positive strategy periods"),
        ]
    )
    render_spacer("xl")

    render_hero_chart_panel_start()
    render_section_header(
        "Performance Comparison",
        "Cumulative strategy return versus buy-and-hold benchmark under current regime exposure settings.",
    )
    render_chart_frame_start()
    st.plotly_chart(
        build_strategy_comparison_figure(strategy_frame),
        use_container_width=True,
        key="strategy_desk_comparison",
    )
    render_chart_frame_end()
    render_panel_end()

    render_spacer()
    exposure_col, summary_col = st.columns([1.2, 1.0], gap="large")
    with exposure_col:
        render_panel_start()
        render_section_header(
            "Exposure Signal",
            "Regime-driven exposure path over time showing active risk-on and risk-off positioning.",
        )
        render_chart_frame_start()
        st.plotly_chart(
            build_strategy_exposure_figure(strategy_frame),
            use_container_width=True,
            key="strategy_desk_exposure",
        )
        render_chart_frame_end()
        render_panel_end()
    with summary_col:
        render_panel_start()
        render_section_header(
            "Strategy Summary",
            "Regime-level participation and return contribution under the selected exposure rules.",
        )
        summary_table = summarize_strategy_by_regime(strategy_frame)
        st.dataframe(summary_table, use_container_width=True, hide_index=True, height=360)
        render_panel_end()

    render_spacer()
    risk_col, contribution_col = st.columns([1.05, 0.95], gap="large")
    with risk_col:
        render_panel_start()
        render_section_header(
            "Risk Trace",
            "Drawdown profile of the regime-aware strategy to highlight stress periods and recovery depth.",
        )
        render_chart_frame_start()
        st.plotly_chart(
            build_drawdown_figure(strategy_frame),
            use_container_width=True,
            key="strategy_desk_drawdown",
        )
        render_chart_frame_end()
        render_panel_end()
    with contribution_col:
        render_panel_start()
        render_section_header(
            "State Contribution",
            "Cumulative contribution by semantic state, showing where the strategy earns or gives back performance.",
        )
        render_chart_frame_start()
        st.plotly_chart(
            build_regime_return_contribution_figure(strategy_frame),
            use_container_width=True,
            key="strategy_desk_state_contribution",
        )
        render_chart_frame_end()
        render_panel_end()


def render_about_page() -> None:
    """Render the about page."""
    render_page_header(
        "System",
        "About",
        "A unified financial analytics workspace that combines machine learning workflow rigor with quant-style market visualization.",
    )
    render_panel_start()
    render_section_header(
        "Platform Overview",
        "Designed as an institutional-style analytics workspace where machine learning diagnostics and quantitative market views share the same visual system.",
    )
    st.write(
        "This project combines an academic machine learning pipeline with a professional quantitative analytics workspace for unsupervised market regime detection."
    )
    st.write(
        "The workflow is adapted specifically for financial time-series data, integrating input-data setup, EDA, cleaning, feature engineering, time-based splitting, clustering validation, hyperparameter tuning, and chart-led quantitative review in one reusable dashboard."
    )
    render_panel_end()


def main() -> None:
    configure_page()
    init_session_state()
    scroll_page_to_top_if_needed()
    if st.session_state.current_page == "Landing":
        render_landing_page()
        return

    page = render_sidebar()
    render_topbar(page)

    if page == "Landing":
        render_landing_page()
    elif page == "Dataset Overview":
        render_dataset_overview_page()
    elif page == "ML Pipeline":
        render_ml_pipeline_page()
    elif page == "Market Regime Analysis":
        render_market_regime_analysis_page()
    elif page == "Trading Strategy":
        render_trading_strategy_page()
    elif page == "Correlation Matrix":
        render_correlation_matrix_page()
    elif page == "Volatility Analysis":
        render_volatility_analysis_page()
    elif page == "Returns Analysis":
        render_returns_analysis_page()
    else:
        render_about_page()


if __name__ == "__main__":
    main()
