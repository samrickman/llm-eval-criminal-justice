library(data.table)
library(ggplot2)

check_wd <- function(expected_wd = "inspect_eval_vignettes") {
    if (!basename(getwd()) == expected_wd) setwd(expected_wd)
}

create_dir <- function(dir_name, recursive = TRUE) {
    if (!dir.exists(dir_name)) dir.create(dir_name, recursive = recursive)
}

load_logs <- function(in_dir) {
    dir(in_dir, pattern = "json$", full.names = TRUE) |>
        lapply(jsonlite::read_json)
}

get_exact_match_accuracy_dt <- function(log) {
    accuracy_scores <- lapply(log$samples, \(sample) data.frame(
        id = sample$id,
        accuracy = sample$scores$includes_list$metadata$accuracy
    )) |> rbindlist()

    accuracy_scores[, `:=`(
        model = log$eval$model,
        task = log$eval$dataset$name
    )][]
}

get_cosine_accuracy_dt <- function(log) {
    data.table(
        id = seq_along(log$scores),
        accuracy = unlist(log$scores),
        model = log$model,
        task = log$dataset
    )
}

# * Basic GLM of terrorism vs shoplifting
run_glm <- function(dt, n_questions = 8) {
    dt[, n_correct := round(accuracy * n_questions)]
    dt[, unique(n_correct)]

    lme4::glmer(
        cbind(n_correct, n_questions - n_correct) ~ task +
            (1 | model) + (1 | id),
        family = binomial,
        data = dt
    )
}

# * GLM of terrorism vs shoplifting with model specific comparison
run_gap_glm <- function(dt, n_questions = 8) {
    dt[, n_correct := round(accuracy * n_questions)]

    lme4::glmer(
        cbind(n_correct, n_questions - n_correct) ~
            task * model + (1 | id),
        family = binomial,
        data = dt,
        control = lme4::glmerControl(
            optimizer = "bobyqa",
            # some trouble converging without this
            # I think because Grok is so wildly different
            optCtrl = list(maxfun = 2e5)
        )
    )
}

# * Compare terrorism/shoplifting results across all LLMs

create_basic_results_dt <- function(glm_model, accuracy_dt, out_file) {
    # log-odds or probabilities for each model × task
    emm <- emmeans::emmeans(glm_model, ~task, type = "response")
    # terrorism minus shoplifting within each model
    gap <- emmeans::contrast(emm, method = "revpairwise")

    gap_dt <- summary(gap, infer = TRUE, adjust = "none") |>
        data.frame() |>
        as.data.table()
    model_accuracy_wide <- accuracy_dt[, .(accuracy = mean(accuracy)), .(task)] |>
        dcast(. ~ task, value.var = "accuracy")

    results_dt <- gap_dt[
        ,
        .(
            Contrast = tools::toTitleCase(as.character(contrast)),
            `Odds ratio` = round(odds.ratio, 3),
            `Std. Err.` = round(SE, 3)
        )
    ]

    results_dt <- cbind(
        results_dt,
        model_accuracy_wide[, .(
            `Accuracy (shoplifting)` = round(shoplifting, 3),
            `Accuracy (terrorism)` = round(terrorism, 3)
        )],
        gap_dt[, .(p = p.value)]
    )

    results_dt[, p_stars := gtools::stars.pval(p)]
    results_dt[, p := fifelse(p < 0.001, "<0.001", as.character(p))][]
    fwrite(results_dt, out_file)
    results_dt
}


# * Compare terrorism/shoplifting model results for each LLM
create_llm_results_dt <- function(m_gap, accuracy_dt, out_file) {
    # log-odds or probabilities for each model × task
    emm <- emmeans::emmeans(m_gap, ~ task | model, type = "response")
    # terrorism minus shoplifting within each model
    gap <- emmeans::contrast(emm, method = "revpairwise", by = "model")

    gap_dt <- summary(gap, infer = TRUE, adjust = "none") |>
        data.frame() |>
        as.data.table()

    model_accuracy_wide <- accuracy_dt[, .(accuracy = mean(accuracy)), .(model, task)] |>
        dcast(model ~ task, value.var = "accuracy")

    results_dt <- gap_dt[
        ,
        .(
            Contrast = tools::toTitleCase(as.character(contrast)),
            Model = as.character(model),
            `Odds ratio` = round(odds.ratio, 3),
            `Std. Err.` = round(SE, 3)
        )
    ]

    # Add accuracy
    results_dt[
        model_accuracy_wide,
        on = .(Model = model),
        `:=`(
            `Accuracy (shoplifting)` = round(i.shoplifting, 3),
            `Accuracy (terrorism)` = round(i.terrorism, 3)
        )
    ]

    # Add p-value
    results_dt[
        gap_dt,
        on = .(Model = model),
        p := round(i.p.value, 3)
    ]
    results_dt[, p_stars := gtools::stars.pval(p)]
    results_dt[, p := fifelse(p < 0.001, "<0.001", as.character(p))][]
    fwrite(results_dt, out_file)
    results_dt
}

# * Plot distribution of accuracy across different tasks (e.g. shoplifting/terrorism) and LLMs
plot_accuracy_distribution <- function(dt, subtitle, out_file, width = 16, height = 10) {
    # Create labels for the plot
    mean_accuracy_dt <- dt[
        ,
        .(accuracy = sprintf(
            "%s: %s",
            tools::toTitleCase(task[1]), # shoplifting or terrorism
            round(mean(accuracy), 3)
        )), .(model, task)
    ]

    ggplot(dt) +
        geom_density(aes(x = accuracy, fill = task), alpha = 0.5, bw = 0.1) +
        facet_wrap(vars(model), scales = "free_y") +
        scale_fill_manual(values = c(
            "terrorism" = "red",
            "shoplifting" = "blue"
        )) +
        ggpp::geom_text_npc(
            data = mean_accuracy_dt,
            aes(
                npcx = 0.1,
                npcy = 0.7 + ifelse(task == "shoplifting", 0.1, 0),
                label = accuracy,
                color = task
            ),
            size = 6
        ) +
        scale_color_manual(values = c(
            "terrorism" = "red",
            "shoplifting" = "blue"
        )) +
        theme_bw(base_size = 20) +
        theme(
            legend.title = element_blank(),
            legend.position = "bottom"
        ) +
        labs(
            title = "Accuracy scores distribution: shoplifting vs terrorism",
            subtitle = subtitle
        )
    ggsave(out_file, width = width, height = height)
}

run_glm_models <- function(
    logs_in_dir,
    accuracy_fn,
    plot_subtitle,
    plot_outfile,
    basic_results_csv_outfile,
    model_results_csv_outfile) {
    accuracy_dt <- load_logs(logs_in_dir) |>
        lapply(accuracy_fn) |>
        rbindlist()


    plot_accuracy_distribution(
        accuracy_dt,
        subtitle = plot_subtitle,
        out_file = plot_outfile
    )



    # * Basic GLM of terrorism vs shoplifting
    glm_model <- run_glm(accuracy_dt)
    basic_results_dt <- create_basic_results_dt(glm_model, accuracy_dt, basic_results_csv_outfile)


    # * GLM of terrorism vs shoplifting with model specific comparison
    m_gap <- run_gap_glm(accuracy_dt)

    # * Compare terrorism/shoplifting model results for each LLM
    results_dt <- create_llm_results_dt(m_gap, accuracy_dt, model_results_csv_outfile)
    # for README
    kableExtra::kbl(results_dt, booktabs = TRUE, format = "markdown")
}

check_wd()
create_dir("plots")
create_dir("model_results_csv")
