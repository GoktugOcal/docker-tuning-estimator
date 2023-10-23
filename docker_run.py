import docker
import os

client = docker.from_env()

container_name = "data1"
image_name = "trainer"

log_file = "./logs/logz.txt"

with open(log_file, "w") as f:
    f.write("cpus,batch_size,total_batches,no_params,epoch_no,duration\n")

for cores in [2,4,8]:
    for batch_size in [32,64,128]:
        for model_no in [0,5,9]:

            model_path = f"./models/iter_{model_no}_best_model.pth.tar"

            command = [
                "python", "trainer.py",
                "--cpus", str(cores),
                "--batch_size", str(batch_size),
                "--model_path", model_path,
                "--log_path", log_file]

            volumes = {os.getcwd() + "/logs": {'bind': "/app/logs", 'mode': 'rw'}}

            container = client.containers.run(
                image=image_name,
                command=command,
                name=container_name,
                nano_cpus= cores*1000000000,
                mem_limit="4g",
                remove=True,  # Automatically remove the container when it stops
                detach=False,
                volumes=volumes
            )

            container.remove()