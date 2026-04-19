import jinja2
import argparse
import os


def render_build_file(op_name, output_path="./"):
    env = jinja2.Environment(
        loader=jinja2.PackageLoader("mirai", "templates"),
        undefined=jinja2.StrictUndefined,
    )
    tpl = env.get_template("build.sh.tpl")
    result = tpl.render(kernels=op_name)
    with open(os.path.join(output_path, f"build.sh"), "w", encoding="utf-8") as f:
        f.write(result)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str, help="model name to be set", nargs="+")
    parser.add_argument("--output_path", type=str, help="output path of generated code", default="./")

    args = parser.parse_args()
    op_name = args.model_name
    output_path = args.output_path

    render_build_file(op_name, output_path)


if __name__ == "__main__":
    main()
