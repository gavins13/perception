


class Execution:
    - load Config.perception file
    - load Experiments.perception file
    - Checkpointing, restoration, writing the saves to files
    - Keeps track of experiments by writing to a experiments txt file
    - Loads tensorboard during execution using port number specified in the exp config using no GPUs and 


class Execution(object):
    def training(self):
        '''
        Training Loop
        '''
        #global_step = tf.train.get_or_create_global_step()
        step = 0
        for epoch in range(max_epochs):

        for images in train_dataset:
            images = tf.image.resize(images,[32,32])
            batch_size=images.shape[0]
            start_time = time.time()

            # generating noise from a uniform distribution
            noise = tf.random.normal([batch_size, 1, 1, noise_dim])
            rotation_n = tf.random.uniform([], minval=0, maxval=3, dtype=tf.dtypes.int32, seed=operation_seed)
            if second_unpaired is True:
                noise_2 = tf.random.normal([batch_size, 1, 1, noise_dim])
            rotation = tf.cast(rotation_n, dtype=tf.float32) * np.pi/2.

            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape, tf.GradientTape() as disc_rot_tape, tf.GradientTape() as disc_2_tape:
            generated_images = generator(noise, training=True)
            #print('.',images.shape)
            images_rot = tf.image.rot90(images, k=rotation_n)
            generated_images_rot = tf.image.rot90(generated_images, k=rotation_n)

            real_logits = discriminator(images, training=True)
            fake_logits = discriminator(generated_images, training=True)
            real_logits_rot = discriminator(images_rot, training=True, predict_rotation=True)
            fake_logits_rot = discriminator(generated_images_rot, training=True, predict_rotation=True)

            if second_unpaired is True:
                generated_images_2 = generator(noise_2, training=True)
                fake_logits_2 = discriminator(generated_images_2, training=True)
                disc_loss_2 = discriminator_loss(real_logits, fake_logits_2, rotation_n, real_logits_rot) # [] CHECK

            gen_loss = generator_loss(fake_logits, rotation_n, fake_logits_rot)
            disc_loss = discriminator_loss(real_logits, fake_logits, rotation_n, real_logits_rot)

            gradients_of_generator = gen_tape.gradient(gen_loss, generator.variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.variables)
            # gradients_of_discriminator_rot = disc_rot_tape.gradient(disc_loss_rot, discriminator.variables)

            generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.variables))
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.variables))
            # discriminator_rot_optimizer.apply_gradients(zip(gradients_of_discriminator_rot, discriminator.variables))

            if second_unpaired is True:
                gradients_of_discriminator_2 = disc_2_tape.gradient(disc_loss_2, discriminator.variables)
                discriminator_optimizer_2.apply_gradients(zip(gradients_of_discriminator_2, discriminator.variables))

            epochs = step * batch_size / float(len(train_data))
            duration = time.time() - start_time
            if step % print_steps == 0:
            display.clear_output(wait=True)
            examples_per_sec = batch_size / float(duration)
            #print("Epochs: {:.2f} global_step: {} loss_D: {:.3f} loss_G: {:.3f} ({:.2f} examples/sec; {:.3f} sec/batch)".format(
            #          epochs,
            #          step,
            #          np.mean(disc_loss),
            #          np.mean(gen_loss),
            #          examples_per_sec,
            #          duration))
            sample_data = generator(random_vector_for_generation, training=False)
            print_or_save_sample_images(sample_data.numpy())

            step += 1
            print("%d         " % (step), end="\r")

            #if epoch % 1 == 0:
            if step % 20 == 0:
                images = tf.tile(images, [16,1,1,1])
                images_rot = tf.tile(images, [16,1,1,1])
                print("%d  saving" % (step), end="\r")
                display.clear_output(wait=True)
                print("This images are saved at {} epoch".format(epoch+1))
                sample_data = generator(random_vector_for_generation, training=False)
                print_or_save_sample_images(sample_data.numpy(), is_save=True, epoch=epoch+1)
                print_or_save_sample_images(images.numpy(), is_save=True, epoch=epoch+1, prefix="REAL_")
                print_or_save_sample_images(images_rot.numpy(), is_save=True, epoch=epoch+1, prefix="REAL_TRANSFORMED_")

        # saving (checkpoint) the model every save_epochs
        if (epoch + 1) % save_epochs == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
            

        # Save full batch of training images to calculate FID
        if (epoch + 1) % full_save_epochs == 0:
            for batch in range(full_save_num_images // batch_size):
            random_vector_for_full_save = tf.random.normal([batch_size, 1, 1, noise_dim])
            sample_data = generator(random_vector_for_full_save, training=False)
            sample_blob = sample_data.numpy()
            h5f = h5py.File('saved/imageblob_epoch{}_{}.h5'.format(epoch+1, batch+1), 'w')
            h5f.create_dataset('imageblob', data=sample_blob)
            h5f.close()